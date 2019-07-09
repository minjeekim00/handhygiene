import torch
import torch.utils.data as data
from torchvision.datasets.folder import find_classes
from .i3ddataset import * # get_framepaths, get_flownames
from .makedataset import make_hh_dataset
from .makedataset import target_dataframe
from .makedataset import get_keypoints

import os
import numpy as np
from PIL import Image
from tqdm import tqdm
from glob import glob
import logging

import sys
sys.path.append('./utils/python-opencv-cuda/python')
import common as cm

def make_dataset(dir, class_to_idx, df, data):
    exclusions = ['38_20190119_frames000643', 
                  '40_20190208_frames026493',
                  '34_20190110_frames060785', #window
                  '34_20190110_frames066161',
                  '34_20190110_frames111213']
    fnames, coords, labels = make_hh_dataset(dir, class_to_idx, df, data, exclusions)
    targets = labels_to_idx(labels)
    return [fnames, coords, targets]


def default_loader(fnames, coords):
    if any('38_20190119_frames000643' in fname for fname in fnames):
        print(fnames)
    rgbs = video_loader(fnames, coords)
    flows = optflow_loader(fnames, coords, False)
    return rgbs, flows


def video_loader(fnames, coords):
    """
        return: list of PIL Images
    """
    video = []
    for i, fname in enumerate(fnames):
        with open(fname, 'rb') as f:
            img = Image.open(f)
            img = img.convert('RGB')
            video.append(img)
    return video


def get_flowdir(dir, reversed=False, cropped=True):
    if cropped:
        name='cropped_flow' if not reversed else 'cropped_reverse_flow'
    else:
        name='flow' if not reversed else 'reverse_flow'
    flowdir=os.path.join(dir, name)
    return flowdir


def get_flownames(fnames, reversed=False, cropped=True):
    ffnames=[]
    for img in fnames:
        dir = os.path.split(img)[0]
        tail = os.path.split(img)[1]
        name, ext = os.path.splitext(tail)
        if len(dir.split('_'))>3: # for augmentated dir
            start = int(dir.split('_')[-1])
            dir = dir.replace('_{}'.format(start), '')
        flowdir = get_flowdir(dir, reversed, cropped)
        flow = os.path.join(flowdir, name+'_flow'+ext)
        ffnames.append(flow)
    return ffnames


def optflow_loader(fnames, coords, cropped=True):
    """
        return: list of PIL Images
    """
    isReversed=check_reverse(fnames)
    ffnames = get_flownames(fnames, isReversed, cropped)
    if any(not os.path.exists(f) for f in ffnames):
        dir = os.path.split(fnames[0])[0]
        #cal_for_frames(dir)
        cm.findOpticalFlow(dir, True, True, isReversed, cropped)
        
    return video_loader(ffnames, coords)


def check_reverse(frames):
    return True if frames != sorted(frames) else False

    
class HandHygiene(I3DDataset):
        
    def __init__(self, root, split='train', clip_len=16, 
                 spatial_transform=None,
                 temporal_transform=None,
                 openpose_transform=None,
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
        self.openpose_transform = openpose_transform
        ## check optical flow
        if preprocess:
            self.preprocess(num_workers)
            
            
    def __getitem__(self, index):
        # loading and preprocessing.
        fnames= self.samples[0][index]
        findices = get_framepaths(fnames)
        coords= self.samples[1][index]
        
        logging.info("sample: {}".format(index))
        if self.temporal_transform is not None:
            findices, coords = self.temporal_transform(findices, coords)
        clips, flows = self.loader(findices, coords)
        
        if self.openpose_transform is not None:
            self.openpose_transform.randomize_parameters()
            streams = [self.openpose_transform(img, flow, coords, i)
                       for i, (img, flow) in enumerate(zip(clips, flows))]
            if len(streams[0])==0:
                print(self.__getpath__(index), "windows empty")
                print(coords)
            clips = [stream[0] for stream in streams]
            flows = [stream[1] for stream in streams]
        
        if self.spatial_transform is not None:
            self.spatial_transform.randomize_parameters()
            clips = [self.spatial_transform(img) for img in clips]
            flows = [self.spatial_transform(img) for img in flows]
        
        target = self.samples[2][index]
        if self.target_transform is not None:
            target = self.target_transform(target)
        
        target = torch.tensor(target).unsqueeze(0)
        clips = torch.stack(clips).permute(1, 0, 2, 3)
        flows = [flow[:-1,:,:] for flow in flows]
        flows = torch.stack(flows).permute(1, 0, 2, 3)
        
        return clips, flows, target
        #return clips, flows, target, coords
        
        
    def preprocess(self, num_workers):
        useCuda=True
        if not useCuda:
            from multiprocessing import Pool
            paths = [self.__getpath__(i) for i in range(self.__len__()) 
                     if check_cropped_dir(self.__getpath__(i))]
            pool = Pool(num_workers)
            pool.map(cal_for_frames, paths)
            pool.map(cal_reverse, paths)
            return
        else:
            for path in tqdm(paths):
                cm.findOpticalFlow(path, useCuda, True, False, True)