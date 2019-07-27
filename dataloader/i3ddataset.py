import torch
import torch.utils.data as data
from .videodataset import * # TODO: explicit functions
# get_framepaths, video_loader
import sys
sys.path.append('./utils/python-opencv-cuda/python')
import common as cm

import os
import numpy as np
from PIL import Image
from tqdm import tqdm
from glob import glob
import json
import pandas as pd


def default_loader(fnames):
    rgbs = video_loader(fnames)
    flows = optflow_loader(fnames)
    return rgbs, flows

def optflow_loader(fnames):
    ffnames = get_flownames(fnames)
    if any(not os.path.exists(f) for f in ffnames):
        dir = os.path.split(fnames[0])[0]
        cm.findOpticalFlow(dir, True, True, False, False)
    return video_loader(ffnames)

def get_flownames(fnames, reversed=False):
    ffnames=[]
    for img in fnames:
        dir = os.path.split(img)[0]
        tail = os.path.split(img)[1]
        name, ext = os.path.splitext(tail)
        if check_cropped_dir(dir): # remove last _
            dir = '_'.join(dir.split('_')[:-1])
        
        flowdirname='flow' if not reversed else 'reverse_flow'
        flow = os.path.join(dir, flowdirname, name+'_flow'+ext)
        ffnames.append(flow)
    return ffnames

def check_cropped_dir(dir):
    """ to check if the images in dir are temporally cropped from original data"""
    basename = os.path.basename(dir)
    return True if len(basename.split('_'))>3 else False

class I3DDataset(VideoFolder):
        
    def __init__(self, root, split='train', clip_len=16, 
                 spatial_transform=None,
                 temporal_transform=None,
                 target_transform=None,
                 preprocess=False, loader=default_loader, num_workers=1):
        super(I3DDataset, self).__init__(root, split, clip_len,
                                         spatial_transform=spatial_transform,
                                         temporal_transform=temporal_transform,
                                         target_transform=target_transform,
                                         preprocess=preprocess, loader=loader)
        self.root = root
        self.loader = loader
        self.num_workers = num_workers

        if preprocess:
            self.preprocess(num_workers)

            
    def __getitem__(self, index):
        # loading and preprocessing.
        fnames= self.samples[0][index]
        findices = get_framepaths(fnames)
        target = self.samples[1][index]
        
        if self.temporal_transform is not None:
            findices = self.temporal_transform(findices)
        clips, flows = self.loader(findices)
         
        if self.spatial_transform is not None:
            self.spatial_transform.randomize_parameters()
            clips = [self.spatial_transform(img) for img in clips]
            flows = [self.spatial_transform(img) for img in flows]
        
        if self.target_transform is not None:
            target = self.target_transform(target)
        target = torch.tensor(target).unsqueeze(0)
       
        clips = torch.stack(clips).permute(1, 0, 2, 3)
        flows = [flow[:-1,:,:] for flow in flows]
        flows = torch.stack(flows).permute(1, 0, 2, 3)
        
        return clips, flows, target
    
    def preprocess(self, num_workers):
        useCuda=True
        paths = [self.__getpath__(i) for i in range(self.__len__()) 
                     if check_cropped_dir(self.__getpath__(i))]
        if not useCuda:
            from multiprocessing import Pool
            from .opticalflow import cal_for_frames
            from .opticalflow import cal_reverse
            pool = Pool(num_workers)
            pool.map(cal_for_frames, paths)
            pool.map(cal_reverse, paths)
            return
        else:
            for path in tqdm(paths):
                cm.findOpticalFlow(path, useCuda, True, False, False)
    
    def __len__(self):
        return len(self.samples[0]) # fnames
    
    def __getpath__(self, index):
        return self.samples[0][index]
