import torch
import torch.utils.data as data
from .videodataset import *
# https://github.com/minjeekim00/pytorch-dataset/blob/master/Video/videodataset.py
from .opticalflow import compute_TVL1
from .opticalflow import cal_for_frames
#https://github.com/minjeekim00/pytorch-dataset/blob/master/Video/preprocessing/opticalflow.py

import os
import numpy as np
from PIL import Image

from tqdm import tqdm
from glob import glob

import json
import pandas as pd


def default_loader(dir):
    rgbs = video_loader(dir)
    flows = optflow_loader(dir)
    return rgbs, flows

def optflow_loader(dir):
    flow = get_flow(dir)
    flows = []
    for i, flw in enumerate(flow):
        shape = flw.shape
        # to make extra 3 channel to use torchvision transform
        tmp = np.empty((shape[0], shape[1], 1)).astype(np.uint8) 
        img = np.dstack((flw.astype(np.uint8), tmp))
        img = Image.fromarray(img)
        flows.append(img)
    return flows

def get_flow(dir):
    """
        return: (1, D, H, W, 2) shape of array of .npy
    """
    basename = os.path.basename(dir)
    
    if len(basename.split('_')) > 3: # when temporal sampling
        start = int(basename.split('_')[-1])
        currbasename = basename.rsplit('_', 1)[0]
        currdir = dir.rsplit('/', 1)[0]
        flow_dir = os.path.join(currdir, currbasename, '{}.npy'.format(currbasename))
        if os.path.exists(flow_dir):
            flows = np.load(flow_dir)
            return flows[start:start+16] ## clip_len
        #else: ## TODO: when base npy not exists
      
    flow_dir = os.path.join(dir,'{}.npy'.format(basename))
    if os.path.exists(flow_dir):
        return np.load(flow_dir)
    flow = cal_for_frames(dir)
    return flow

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
        
        if self.temporal_transform is not None:
            fnames = self.temporal_transform(fnames)
        clips, flows = self.loader(fnames)
         
        if self.spatial_transform is not None:
            clips = [self.spatial_transform(img) for img in clips]
            flows = [self.spatial_transform(img) for img in flows]
        clips = torch.stack(clips).permute(1, 0, 2, 3)
        flows = [flow[:-1,:,:] for flow in flows]
        flows = torch.stack(flows).permute(1, 0, 2, 3)

        targets = self.samples[1][index]
        if self.target_transform is not None:
            target = self.target_transform(target)
            
        #targets = torch.tensor(targets).unsqueeze(0)
        return clips, flows, targets
    
    
    def preprocess(self, num_workers):
        from multiprocessing import Pool
        paths = [self.__getpath__(i) for i in range(self.__len__())]
        pool = Pool(num_workers)
        pool.map(get_flow, paths)
        return
    
    def __len__(self):
        return len(self.samples[0]) # fnames
    
    def __getpath__(self, index):
        return self.samples[0][index]
