import torch
import torch.utils.data as data
from torchvision.datasets.folder import find_classes
from .i3ddataset import *
# https://github.com/minjeekim00/pytorch-dataset/blob/master/Video/I3D/i3ddataset.py
from .opticalflow import compute_TVL1
from .opticalflow import cal_for_frames
#https://github.com/minjeekim00/pytorch-dataset/blob/master/Video/preprocessing/opticalflow.py
from .poseroi import calc_margin
from .poseroi import crop_by_clip

import os
import numpy as np
from PIL import Image

from tqdm import tqdm
from glob import glob

def make_dataset(dir, class_to_idx, df, data):
    """
        fnames: name of directory containing images
        coords: dict containg people, torso coordinates
        labels: class
    """
    np.random.seed(50)
    fnames, coords, labels = [], [], []
    
    exclusions = ['38_20190119_frames000643']
    lists = df['imgpath'].values
    
    for label in os.listdir(os.path.join(dir)):
        for fname in os.listdir(os.path.join(dir, label)):
            if is_image_file(fname):
                continue
            if fname not in lists:
                continue
            if fname in exclusions:
                continue

            frames = sorted([os.path.join(dir, img) for img 
                             in os.listdir(os.path.join(dir, label, fname))])
            frames = [img for img in frames if is_image_file(img)]

                
            item = [row for row in data if row['imgpath']==fname][0]
            people = item['people']
            torso = item['torso']
            npeople = len(item['people'])
            tidxs = df[df['imgpath']==fname]['targets'].values[0] # target idx
            tidxs = [int(t) for t in tidxs.strip().split(',')]
            nidxs = list(range(npeople))
            nidxs = [int(n) for n in nidxs if n not in tidxs]

            
            ## appending clean
            for tidx in tidxs:
                if len(frames) != len(people[tidx]):
                    print("<{}> {} coords and {} frames / of people {}"
                              .format(fname, len(people[tidx]), len(frames), tidx))
                    print(people[tidx])
                    continue
                fnames.append(os.path.join(dir, label, fname))
                coords.append({'people':people[tidx], 'torso':torso[tidx]})
                labels.append(label)
                
            ## appending notclean 
            if len(nidxs) > 0:
                max = np.random.randint(1, 2+1)
                for nidx in nidxs[:max]:
                    if len(frames) != len(people[nidx]):
                        print("<{}> {} coords and {} frames / of people {}"
                                  .format(fname, len(people[nidx]), len(frames), nidx))
                        print(people[nidx])
                        continue

                    fnames.append(os.path.join(dir, label, fname))
                    coords.append({'people':people[nidx], 'torso':torso[nidx]})
                    labels.append('notclean')

    #assert len(labels) == len(fnames)
    print('Number of {} people: {:d}'.format(dir, len(fnames)))
    targets = labels_to_idx(labels)
    
    return [fnames, coords, targets]


def target_dataframe(path='./data/label.csv'):
    
    import pandas as pd
    df=pd.read_csv(path)
     # target 있는 imgpath만 선별
    df = df[df['targets'].notnull()]
    return df

def get_keypoints(path='./data/keypoints.txt'):
    
    import json
    with open(path, 'r') as file:
        data = json.load(file)
        data = data['coord']
    return data


def default_loader(fnames, coords):
    rgbs = video_loader(fnames, coords)
    flows = optflow_loader(fnames, coords)
    return rgbs, flows
 
def crop_pil_image(coords, idx):
    people = coords['people']
    torso = coords['torso']
    
    try:
        window = people[idx]
        if window is not None:
            window = calc_margin(torso, window)
        else:
            window = (0, 0, 0, 0)
    except:
        print("{} fail to calculate margin".format(idx))
        window = None
    return window


def video_loader(frames, coords):
    """
        return: list of PIL Images
    """
    video = []
    cropped = [] # coordinates
    for i, fname in enumerate(frames):
        # calc margin
        window = crop_pil_image(coords, i)
        cropped.append(window)
        with open(fname, 'rb') as f:
            img = Image.open(f)
            img = img.convert('RGB')
            video.append(img)
    
    video = crop_by_clip(video, cropped)
    return video


def optflow_loader(fnames, coords):
    """
        return: list of PIL Images
    """
    ffnames = get_flownames(fnames)
    if any(not os.path.exists(f) for f in ffnames):
        dir = os.path.split(fnames[0])[0]
        cal_for_frames(dir)
        
    return video_loader(ffnames, coords)

    
class HandHygiene(I3DDataset):
        
    def __init__(self, root, split='train', clip_len=16, 
                 spatial_transform=None,
                 temporal_transform=None,
                 target_transform=None,
                 preprocess=False, loader=default_loader, num_workers=1):

        super(HandHygiene, self).__init__(root, split, clip_len,
                                         spatial_transform=spatial_transform,
                                         temporal_transform=temporal_transform,
                                         target_transform=target_transform,
                                         preprocess=preprocess, loader=loader,
                                         num_workers=num_workers)
        
        df = target_dataframe()
        keypoints = get_keypoints()
        folder = os.path.join(self.image_dir, split)
        classes, class_to_idx = find_classes(folder)
        
        self.loader = loader
        self.samples = make_dataset(folder, class_to_idx, df, keypoints)
        ## check optical flow
        if preprocess:
            self.preprocess(num_workers)
            
            
    def __getitem__(self, index):
        # loading and preprocessing.
        fnames= self.samples[0][index]
        findices = get_framepaths(fnames)
        coords= self.samples[1][index]
        
        if self.temporal_transform is not None:
            findices = self.temporal_transform(findices)
        clips, flows = self.loader(findices, coords)
         
        if self.spatial_transform is not None:
            clips = [self.spatial_transform(img) for img in clips]
            flows = [self.spatial_transform(img) for img in flows]
        clips = torch.stack(clips).permute(1, 0, 2, 3)
        flows = [flow[:-1,:,:] for flow in flows]
        flows = torch.stack(flows).permute(1, 0, 2, 3)

        targets = self.samples[2][index]
        if self.target_transform is not None:
            target = self.target_transform(target)
            
        targets = torch.tensor(targets).unsqueeze(0)
        return clips, flows, targets

    