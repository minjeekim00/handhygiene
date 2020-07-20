import os
import re
import gc
import json
import torch
import numpy as np
import pandas as pd
from glob import glob
from PIL import Image


def get_frames(dirname):
    from torchvision.datasets.folder import is_image_file
    return sorted([os.path.join(dirname, file) 
                   for file in os.listdir(dirname) 
                   if is_image_file(file)])

def read_video(dirname, start_pts=0, end_pts=None, has_bbox=False, downsample=None):
    frames = get_frames(dirname)
    video = []
    
    w = 0
    h = 0
    for i, frame in enumerate(frames):
        with open(frame, 'rb') as f:
            img = Image.open(f)
            img = img.convert('RGB')
            
            if downsample:
                w, h = np.asarray(img).shape[:2]
                ratio = min(w/downsample, h/downsample)
                w /= ratio
                h /= ratio
                img.thumbnail((h, w), Image.ANTIALIAS)
                #img.save(frame, "PNG")
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
    info = {'width': w,
            'height': h,
            'video_fps': 15.0,
           'keypoints': None,
           'annotations': []}
    
    ## when has detection box in image frame
    '''
    if has_bbox:
        fname = os.path.basename(dirname)
        txtfile = os.path.join(dirname, '{}.txt'.format(fname))
        with open(txtfile, 'r') as f:
            item = json.load(f)
        
        ## this is to get body keypoint coordinates,
        ## otherwise skip
        if not True: #need_preprocess:
            info['keypoints'] = item
        else:
            coords = preprocess_keypoints(dirname, item)
            info['keypoints'] = coords

        condition = len(coords['people']) == len(frames)
        message = "{}: frames:{}, coords:{}".format(
            dirname, len(frames), len(coords['people']))
        assert condition, message
    '''
    """
    if has_bbox:
        from natsort import natsorted
        fname = os.path.basename(dirname)
        jsonfiles = natsorted(glob(os.path.join(dirname, '*.json')))
        
        for jsonfile in jsonfiles:
            '''
            bboxes = {{person_id_1: [[x1, y1, x2, y2], [label]] 
                       person_id_2: [[x1, y1, x2, y2], [label]]}}
            '''
            bboxes = _get_bounding_box(fname, jsonfile, bbox_type='labelme')
            info['annotations'].append(bboxes) 
            
            # TODO: confidence score thresholding
    """
            
    if has_bbox:
        for frame in frames:
            image_path = os.path.splitext(os.path.basename(frame))[0]
            bboxes = _get_bbox_info(image_path)
            info['annotations'].append(bboxes)

    #info['annotations'] = _get_completed_annotations(info['annotations'])
    sample = (video, audio, info)
    return read_video_as_clip(sample, start_pts, end_pts, has_bbox)

    
def target_dataframe(path='./data/annotations/hh_target.csv'):
    import pandas as pd
    df = pd.read_csv(path)
    df = df[df['person_id'].notnull()] # target 있는 imgpath만 선별
    return df

    
def read_video_as_clip(sample, start_pts, end_pts, has_bbox, target=None):
    video, audio, info = sample
    video = video[start_pts:end_pts+1]
    
    if end_pts == len(video):
        return (video, audio, info)
    
    if has_bbox:
        ##TODO: slicing keypoints
        #info['keypoints'] = info['keypoints'][start_pts:end_pts+1]
        info_tmp = info.copy()
        info_tmp['annotations'] = info_tmp['annotations'][start_pts:end_pts+1]
        
        if target is not None:
            info_tmp['annotations'] = _get_target_annotations(info_tmp['annotations'], target)
        
        #info_tmp['annotations'] = _get_completed_annotations(info_tmp['annotations'])
        return (video, audio, info_tmp)
    
    return (video, audio, info)


def read_video_timestamps(dirname):
    """ tmp function """
    frames = get_frames(dirname)
    return (list(range(len(frames)*1)), 15.0)


def _get_bbox_info(image_path, df=target_dataframe()):
    
    if 'flow' in image_path:
        image_path = image_path.replace('_flow', '')
    rows = df[df['image_path']==image_path]
    assert len(rows) > 0, image_path
    
    box_dict = {}
    for row in rows.values:
        _, _, label, person_id, x1, y1, x2, y2 = row
        box_dict[person_id] = [[x1, y1, x2, y2], label]
    return box_dict

######################################################################################################


def _get_action_name(fname, person_id, df=target_dataframe()):
    """ return action label from data frame """
    row = df[(df['clip_name'] == fname) & (df['person_id'] == person_id)]
    assert len(row) <= 1, "There is(are) {} labels on person {} in {}".format(len(row), person_id, fname)
    
    if len(row) == 1:
        # TODO:
        if row['action_name'].values[0] == 'touching_patient':
            return 'other_action'
        else:
            return row['action_name'].values[0]
    else: # no label
        return 'other_action'
    
    
def _get_bounding_box(fname, jsonfile, bbox_type='labelme'):
    ''' feeding action label here '''
    
    import json
    if bbox_type == 'labelme':
        bboxes = {}
        
        with open(jsonfile, 'r') as f:
            item = json.load(f)

            assert os.path.splitext(item['imagePath'])[0] == os.path.basename(os.path.splitext(jsonfile)[0])

            for person in item['shapes']:
                person_id = person['group_id']
                bbox = list(np.array(person['points']).reshape((4)))

                if person_id not in bboxes:
                    label = _get_action_name(fname, person_id)
                    bboxes[person_id] = [bbox, label]
    #TODO: else
    return bboxes


def _get_person_ids(annotations):
    person_ids = []
    for ann in annotations:
        for person_id in ann.keys():
            if person_id not in person_ids:
                person_ids.append(person_id)
    return sorted(person_ids)

def _get_completed_annotations(annotations):
    """Grab only completed annotations throughout the video clip"""
    person_ids = _get_person_ids(annotations)
    for person_id in person_ids:
        
        if all([person_id in ann.keys() for ann in annotations]):
            continue
        
        # pop from each annotations
        for ann in annotations:
            if person_id in ann.keys():
                ann.pop(person_id)
    return annotations

def _get_target_annotations(annotations, target):
    """Grab only completed annotations throughout the video clip"""
    
    info_new = []
    for ann in annotations:
        ann_new = {}
        for person_id in ann.keys():
            if person_id == target:
                ann_new[person_id] = ann[person_id]
        info_new.append(ann_new)
    return info_new

'''
def create_new_dataframe(dirname):
    import json
    from natsort import natsorted
    import pandas as pd
    
    info = {'annotations': []}
    
    fname = os.path.basename(dirname)
    jsonfiles = natsorted(glob(os.path.join(dirname, '*.json')))

    for jsonfile in jsonfiles:
        bboxes = _get_bounding_box(fname, jsonfile, bbox_type='labelme')
        info['annotations'].append(bboxes)
        
    info['annotations'] = _get_completed_annotations(info['annotations'])
    
    vids = []
    frame_names = []
    labels = []
    peson_ids = []
    x1s = []
    y1s = []
    x2s = []
    y2s = []

    for phase in ['train', 'val', 'test']:

        for dirname in tqdm(glob('./data/images_new/{}/*'.format(phase))[:]):

            info = read_video(dirname, has_bbox=True)
            fname = os.path.basename(dirname)
            vid = int(fname.split('_')[0])
            jsonfiles = natsorted(glob(os.path.join(dirname, '*.json')))

            for aidx, ann in enumerate(info['annotations']):
                for key, value in ann.items():
                    person_id = key
                    x1, y1, x2, y2 = value[0]
                    label = value[1]
                    frame_name = os.path.splitext(os.path.basename(jsonfiles[aidx]))[0]

                    vids.append(vid)
                    frame_names.append(frame_name)
                    labels.append(label)
                    peson_ids.append(person_id)
                    x1s.append(x1)
                    y1s.append(y1)
                    x2s.append(x2)
                    y2s.append(y2)
                    
    
    li = [frame_names, labels, peson_ids, x1s, y1s, x2s, y2s]
    df = pd.DataFrame(vids, columns = ['video_id'])
    for cid, column in enumerate(['image_path', 'action', 'person_id', 'x1', 'y1', 'x2', 'y2']):
        df[column] = li[cid]
'''
'''
def preprocess_keypoints(dirname, item, df=target_dataframe()):
    fname = dirname.split('/')[-1]
    label = dirname.split('/')[-2]
    frames = get_frames(dirname)
    
    people = item['people']
    npeople = np.array(people).shape[0]
    torso = item['torso']
    tidxs = df[df['imgpath']==fname]['targets'].values # target idx
    if len(tidxs) == 0:
        print("{} target index not exists".format(dirname))
        return [] 
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
'''