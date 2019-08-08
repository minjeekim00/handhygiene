import os
import re
import gc
import json
import torch
import numpy as np
import pandas as pd
from glob import glob
from PIL import Image
    
from torchvision.datasets.folder import has_file_allowed_extension
from torchvision.datasets.folder import is_image_file
from torchvision.datasets.folder import IMG_EXTENSIONS


def get_frames(dirname):
    return sorted([os.path.join(dirname, file) 
                   for file in os.listdir(dirname) 
                   if is_image_file(file)])

def read_video(dirname, start_pts=0, end_pts=None, has_bbox=False):
    
    frames = get_frames(dirname)
    video = []
    for i, frame in enumerate(frames):
        with open(frame, 'rb') as f:
            img = Image.open(f)
            img = img.convert('RGB')
            img = np.asarray(img)
            video.append(img)
            
    if end_pts is None:
        end_pts = len(video)

    if end_pts < start_pts:
        raise ValueError("end_pts should be larger than start_pts, got "
                         "start_pts={} and end_pts={}".format(start_pts, end_pts))
        
    video = np.asarray(video)
    video = torch.tensor(video)
    audio = torch.tensor([]) #tmp
    info = {'video_fps': 15.0,
           'body_keypoint': None}
    
    ## when has detection box in image frame
    if has_bbox:
        basename = os.path.basename(dirname)
        txtfile = os.path.join(dirname, '{}.txt'.format(basename))
        with open(txtfile, 'r') as f:
            item = json.load(f)
        
        ## this is to get body keypoint coordinates,
        ## otherwise skip
        if not True: #need_preprocess:
            info['body_keypoint'] = item
        else:
            coords = preprocess_keypoints(dirname, item)
            info['body_keypoint'] = coords
    
    sample = (video, audio, info)
    return read_video_as_clip(sample, start_pts, end_pts, has_bbox)

def read_video_as_clip(sample, start_pts, end_pts, has_bbox):
    video, audio, info = sample
    video = video[start_pts:end_pts+1]
    keypoints = info['body_keypoint']
    
    ##TODO: slicing keypoints
    #info['body_keypoint'] = keypoints[start_pts:end_pts+1]
    #print(info)
    return (video, audio, info)


def read_video_timestamps(dirname):
    """ tmp function """
    frames = get_frames(dirname)
    return (list(range(len(frames)*1)), 15.0)


def target_dataframe(path='./data/label.csv'):
    df=pd.read_csv(path)
     # target 있는 imgpath만 선별
    df = df[df['targets'].notnull()]
    return df


def preprocess_keypoints(dirname, item, df=target_dataframe()):
    fname = dirname.split('/')[-1]
    label = dirname.split('/')[-2]
    frames = get_frames(dirname)
    
    people = item['people']
    npeople = np.array(people).shape[0]
    torso = item['torso']
    tidxs = df[df['imgpath']==fname]['targets'].values # target idx
    
    if len(tidxs) == 0:  
        return [] # remove data without label
    else: 
        tidxs = tidxs[0]

    #class = df[df['imgpath']==fname]['class'].values[0] # label 
    tidxs = [int(t) for t in tidxs.strip().split(',')]
    nidxs = list(range(npeople))
    nidxs = [int(n) for n in nidxs if n not in tidxs]
    ## appending clean
    for tidx in tidxs:
        start=0
        end=start+len(frames)

        if len(frames) != len(people[tidx][start:end]):
            print("<{}> coords difference of people {}"
                      .format(fname, len(people[tidx]), len(frames), tidx))
            print(people[tidx])
            continue

        coords = {'people':people[tidx][start:end],
                  'torso':torso[tidx][start:end]}
#         else: ## TODO: change 224 to image shape
#             coords = {'people': [[0, 0, 224, 224] for i in range(start, end)],
#                       'torso':torso[tidx][start:end]}
            
    return coords