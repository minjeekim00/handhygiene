import torch
import torch.utils.data as data
from torchvision.datasets import DatasetFolder
from torchvision.datasets.folder import IMG_EXTENSIONS
from torchvision.datasets.folder import has_file_allowed_extension
from torchvision.transforms import functional as F
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


def is_image_file(filename):
    return has_file_allowed_extension(filename, IMG_EXTENSIONS)

def make_dataset(dir, class_to_idx, df, data):
    exclusions = ['40_20190208_frames026493',
                  '34_20190110_frames060785', #window
                  '34_20190110_frames066161',
                  '34_20190110_frames111213']
    fnames, coords, labels = make_hh_dataset(dir, class_to_idx, df, data, exclusions)
    targets = labels_to_idx(labels)
    return [fnames, coords, targets]

def get_framepaths(fname):
    frames = sorted([os.path.join(fname, img) for img in os.listdir(fname)])
    frames = [img for img in frames if is_image_file(img)]
    return frames
    
def labels_to_idx(labels):
    labels_dict = {label: i for i, label in enumerate(sorted(set(labels)))}
    return np.array([labels_dict[label] for label in labels], dtype=int)

def default_loader(frames):
    return video_loader(frames)
            
def video_loader(frames):
    """
        #return: list of PIL Images
        return: list of numpy array
    """
    video = []
    for i, fname in enumerate(frames):
        with open(fname, 'rb') as f:
            img = Image.open(f)
            img = img.convert('RGB')
            img = np.asarray(img)
            video.append(img)   
    return video


class HandHygiene(DatasetFolder):
        
    def __init__(self, root, split='train', 
                 clip_length_in_frames=16,
                 frames_between_clips=1,
                 spatial_transform=None,
                 temporal_transform=None,
                 openpose_transform=None,
                 target_transform=None,
                 preprocess=False, loader=default_loader, num_workers=1):

        super(HandHygiene, self).__init__(root, loader, IMG_EXTENSIONS,
                                          transform=spatial_transform,
                                          target_transform=target_transform)
        df = target_dataframe()
        keypoints = get_keypoints()
        
        self.loader = loader
        self.video_dir = os.path.join(root, 'videos')
        self.image_dir = os.path.join(root, 'images')
        folder = os.path.join(self.image_dir, split)
        
        classes, class_to_idx = self._find_classes(folder)
        self.classes = classes
        self.samples = make_dataset(folder, class_to_idx, df, keypoints)
        video_list = self.samples[0]
        opflw_list = [os.path.join(x, 'flow') for x in self.samples[0]]
        self.video_clips = self._clips_for_video(video_list,
                                                 clip_length_in_frames,
                                                 frames_between_clips)
        self.opflw_clips = self._clips_for_video(opflw_list,
                                                 clip_length_in_frames,
                                                 frames_between_clips)
        
        self.spatial_transform = spatial_transform
        self.temporal_transform = temporal_transform
        self.target_transform = target_transform
        self.openpose_transform = openpose_transform
        
        ## check optical flow
        if preprocess:
            self.preprocess(num_workers)
    
    def _find_classes(self, dir):
        if sys.version_info >= (3, 5):
            # Faster and available in Python 3.5 and above
            classes = [d.name for d in os.scandir(dir) if d.is_dir()]
        else:
            classes = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
        classes.sort()
        class_to_idx = {classes[i]: i for i in range(len(classes))}
        return classes, class_to_idx
    
    def _clips_for_video(self, video_list, size, step):
        
        def istarget(path):
            return False if 'notclean' in path else True
            
        videos= self._get_videos(video_list)
        video_clips=[]
        for vidx, video in enumerate(videos):
            ### NOT CLEAN CONDITION
            if not istarget(video_list[vidx]):
                step = 4
            clips = self._get_clips(video, size, step)
            for cidx, clip in enumerate(clips):
                video_clips.append((clip, vidx))
                
        print("{} clips from {} videos".format(len(video_clips), len(videos)))
        return video_clips
    
    def _get_videos(self, video_paths):
        videos = []
        for path in video_paths:
            frames = get_framepaths(path)
            video = self.loader(frames)
            videos.append(video)
        return videos
        
    def _get_clips(self, video, size, step):
        """ video: [T H W C]
            return: [num_clips H W C size]
        """
        dim=0
        video_t = torch.tensor(np.asarray(video)) # T HWC
        if len(video_t) < size:
            return video_t.unsqueeze(0)
        video_t = video_t.unfold(dim, size, step) # N HWC T
        video_t = video_t.permute(0, 4, 1, 2, 3)  # N T HWC
        return video_t
    
    
    def __getitem__(self, index):
        # loading and preprocessing.
        
        clip, vidx = self.video_clips[index]
        flow, _ = self.opflw_clips[index]
        coords = self.samples[1][vidx]
        clip = self._to_pil_image(clip)
        flow = self._to_pil_image(flow)
        
        if self.temporal_transform is not None:
            self.temporal_transform.randomize_parameters()
            clip = self.temporal_transform(clip)
            flow = self.temporal_transform(flow)
            
        if self.openpose_transform is not None:
            self.openpose_transform.randomize_parameters()
            streams = [self.openpose_transform(c, f, coords, i)
                       for i, (c, f) in enumerate(zip(clip, flow))]
            if len(streams[0])==0:
                print("windows empty")
            clip = [stream[0] for stream in streams]
            flow = [stream[1] for stream in streams]
            
        if self.spatial_transform is not None:
            self.spatial_transform.randomize_parameters()
            clip = [self.spatial_transform(img) for img in clip]
            flow = [self.spatial_transform(img) for img in flow]
            
        target = self.samples[2][vidx]
        clip = torch.stack(clip).transpose(0, 1) # TCHW-->CTHW
        flow = torch.stack(flow).transpose(0, 1) # TCHW-->CTHW
        flow = flow[:-1,:,:,:] # 3->2 channel
        target = torch.tensor(target).unsqueeze(0) # () -> (1,)
        return clip, flow, target
    
    def _to_pil_image(self, video):
        video = [v.permute(2, 0, 1) for v in video] # for to_pil_image
        return [F.to_pil_image(img) for img in video]
    
    def __len__(self):
        return len(self.video_clips)
    
    def __ref__(self):
        size = self.__len__()
        labels=[]
        for i in range(size):
            clip, vidx = self.video_clips[i]
            label = self.samples[2][vidx]
            labels.append(label)
        
        num_clean = len([l for l in labels if l == 0])
        num_notclean = len([l for l in labels if l == 1])
        print("clean: {}, notclean: {}".format(num_clean, num_notclean))
        