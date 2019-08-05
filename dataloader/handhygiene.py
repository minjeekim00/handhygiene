import torch
import torch.utils.data as data
from torchvision.datasets.utils import list_dir
from torchvision.io.video import write_video
from .i3ddataset import I3DDataset
from .videodataset import make_dataset
from .video_utils import VideoClips
from .makedataset import make_hh_dataset
from .makedataset import target_dataframe
from .makedataset import get_keypoints

import os
import sys
sys.path.append('./utils/python-opencv-cuda/python')
import common as cm
import numpy as np
from PIL import Image
from tqdm import tqdm
from glob import glob
import logging


def labels_to_idx(labels):
    labels_dict = {label: i for i, label in enumerate(sorted(set(labels)))}
    return np.array([labels_dict[label] for label in labels], dtype=int)

def make_dataset(dir, class_to_idx, df, data, cropped):
    exclusions = ['40_20190208_frames026493',
                  '34_20190110_frames060785', #window
                  '34_20190110_frames066161',
                  '34_20190110_frames111213']
    fnames, coords, labels = make_hh_dataset(dir, class_to_idx, df, data, exclusions, cropped)
    targets = labels_to_idx(labels)
    return [fnames, coords, targets]

    
class HandHygiene(I3DDataset):
    def __init__(self, root, frames_per_clip, 
                 step_between_clips=1,
                 frame_rate=None,
                 openpose_transform=None,
                 spatial_transform=None,
                 temporal_transform=None,
                 opt_flow_preprocess=False, cropped=False):

        super(HandHygiene, self).__init__(root, frames_per_clip,
                                         step_between_clips=step_between_clips,
                                         frame_rate=frame_rate,
                                         spatial_transform=spatial_transform,
                                         temporal_transform=temporal_transform)
        
        #if cropped:
        #    self.root = os.path.join(self.root, 'cropped')
        df = target_dataframe()
        keypoints = get_keypoints()
        classes = list(sorted(list_dir(root)))
        class_to_idx = {classes[i]: i for i in range(len(classes))}
        self.classes = classes
        self.samples = make_dataset(self.root, class_to_idx, df, keypoints, cropped)
        self.openpose_transform = openpose_transform
        self.cropped = cropped
            
    def __getitem__(self, idx):
        video, _, _, video_idx = self.video_clips.get_clip(idx)
        optflow, _, _, _ = self.optflow_clips.get_clip(idx)
        coords = self.samples[video_idx][1]
        
        video = self._to_pil_image(video)
        optflow = self._to_pil_image(optflow)
        label = self.samples[video_idx][2]
        
        if self.temporal_transform is not None:
            self.temporal_transform.randomize_parameters()
            video = self.temporal_transform(video)
            optflow = self.temporal_transform(optflow)
            
        if self.openpose_transform is not None:
            self.openpose_transform.randomize_parameters()
            streams = [self.openpose_transform(c, f, coords, i)
                       for i, (c, f) in enumerate(zip(video, optflow))]
            if len(streams[0])==0:
                print("windows empty")
            clip = [stream[0] for stream in streams]
            flow = [stream[1] for stream in streams]
            
        
        if self.spatial_transform is not None:
            self.spatial_transform.randomize_parameters()
            video = [self.spatial_transform(img) for img in video]
            optflow = [self.spatial_transform(img) for img in optflow]
            
        video = torch.stack(video).transpose(0, 1) # TCHW-->CTHW
        optflow = torch.stack(optflow).transpose(0, 1) # TCHW-->CTHW
        optflow = optflow[:-1,:,:,:] # 3->2 channel
        label = torch.tensor(label).unsqueeze(0) # () -> (1,)
        return video, optflow, label