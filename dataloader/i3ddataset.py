import torch
import torch.utils.data as data
from .videodataset import VideoFolder
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


class I3DDataset(VideoFolder):
        
    def __init__(self, root, split='train', 
                 clip_length_in_frames=16, 
                 frames_between_clips=1,
                 spatial_transform=None,
                 temporal_transform=None,
                 target_transform=None,
                 preprocess=False, loader=default_loader, num_workers=1):
        super(I3DDataset, self).__init__(root, split,
                                         clip_length_in_frames=clip_length_in_frames,
                                         frames_between_clips=frames_between_clips,
                                         spatial_transform=spatial_transform,
                                         temporal_transform=temporal_transform,
                                         target_transform=target_transform,
                                         preprocess=preprocess, loader=loader)
        
        folder = os.path.join(self.image_dir, split)
        video_list = [x for x in self.samples[0]]
        opflw_list = [os.path.join(x, 'flow') for x in self.samples[0]]
        self.video_clips = self._clips_for_video(video_list,
                                                 clip_length_in_frames,
                                                 frames_between_clips)
        self.opflw_clips = self._clips_for_video(opflw_list,
                                                 clip_length_in_frames,
                                                 frames_between_clips)
        if preprocess:
            self.preprocess(num_workers)

            
    def __getitem__(self, index):
        clip, vidx = self.video_clips[index]
        flow, _ = self.opflw_clips[index]
        clip = self._to_pil_image(clip)
        flow = self._to_pil_image(flow)
        
        if self.temporal_transform is not None:
            self.temporal_transform.randomize_parameters()
            clip = self.temporal_transform(clip)
            flow = self.temporal_transform(flow)
        
        if self.spatial_transform is not None:
            self.spatial_transform.randomize_parameters()
            clip = [self.spatial_transform(img) for img in clip]
            flow = [self.spatial_transform(img) for img in flow]

        target = self.samples[1][vidx]
        clip = torch.stack(clip).transpose(0, 1) # TCHW-->CTHW
        flow = torch.stack(flow).transpose(0, 1) # TCHW-->CTHW
        flow = flow[:-1,:,:,:] # 3->2 channel
        target = torch.tensor(target).unsqueeze(0) # () -> (1,)
        return clip, flow, target
    
    def preprocess(self, num_workers):
        useCuda=True
        paths = self.samples[0]
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