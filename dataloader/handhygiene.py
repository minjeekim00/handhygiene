import torch
import torch.utils.data as data
from torchvision.datasets.folder import find_classes
from .i3ddataset import *
from .opticalflow import compute_TVL1
from .opticalflow import cal_for_frames
from .makedataset import make_hh_dataset
from .makedataset import target_dataframe
from .makedataset import get_keypoints

import os
import numpy as np
from PIL import Image

from tqdm import tqdm
from glob import glob


def make_dataset(dir, class_to_idx, df, data):
    fnames, coords, labels = make_hh_dataset(dir, class_to_idx, df, data)
    targets = labels_to_idx(labels)
    return [fnames, coords, targets]

def default_loader(fnames, coords):
    rgbs = video_loader(fnames, coords)
    flows = optflow_loader(fnames, coords)
    return rgbs, flows

def video_loader(frames, coords):
    """
        return: list of PIL Images
    """
    video = []
    for i, fname in enumerate(frames):
        with open(fname, 'rb') as f:
            img = Image.open(f)
            img = img.convert('RGB')
            video.append(img)
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
        coords_clip = self.samples[1]
        
        if self.temporal_transform is not None:
            findices, coords = self.temporal_transform(findices, coords)
        clips, flows = self.loader(findices, coords)
        
        if self.openpose_transform is not None:
            self.openpose_transform.randomize_parameters()
            clips = [self.openpose_transform(img, coords, i) for i, img in enumerate(clips)]
            flows = [self.openpose_transform(img, coords, i) for i, img in enumerate(flows)]
            
        if self.spatial_transform is not None:
            self.spatial_transform.randomize_parameters()
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

    