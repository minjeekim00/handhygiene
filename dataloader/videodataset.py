import torch
import torch.utils.data as data
from torchvision.datasets import DatasetFolder
from torchvision.datasets.folder import IMG_EXTENSIONS
from torchvision.datasets.folder import has_file_allowed_extension
from torchvision.transforms import functional as F

import os
import sys
import numpy as np
from PIL import Image

from tqdm import tqdm
from glob import glob


def is_image_file(filename):
    return has_file_allowed_extension(filename, IMG_EXTENSIONS)

def make_dataset(dir, class_to_idx):
    fnames, labels = [], []
    for label in sorted(os.listdir(dir)):
        for fname in os.listdir(os.path.join(dir, label)):
            fnames.append(os.path.join(dir, label, fname))
            labels.append(label)
            
    assert len(labels) == len(fnames)
    print('Number of {} videos: {:d}'.format(dir, len(fnames)))
    targets = labels_to_idx(labels)
    return [fnames, targets]

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
            ## todo
            img = np.asarray(img)
            video.append(img)   
    return video


class VideoFolder(DatasetFolder):
    
    def __init__(self, root, split='train', 
                 clip_length_in_frames=16, 
                 frames_between_clips=1,
                 spatial_transform=None,
                 temporal_transform=None,
                 target_transform=None,
                 preprocess=False, loader=default_loader):
        
        super(VideoFolder, self).__init__(root, loader, IMG_EXTENSIONS,
                                          transform=spatial_transform,
                                          target_transform=target_transform)
        self.loader = loader
        self.video_dir = os.path.join(root, 'videos')
        self.image_dir = os.path.join(root, 'images')
        folder = os.path.join(self.image_dir, split)
        
        classes, class_to_idx = self._find_classes(folder)
        self.classes = classes
        self.samples = make_dataset(folder, class_to_idx)
        video_list = [x for x in self.samples[0]]
        self.video_clips = self._clips_for_video(video_list,
                                                 clip_length_in_frames,
                                                 frames_between_clips)
        self.spatial_transform = spatial_transform
        self.temporal_transform = temporal_transform
        self.target_transform = target_transform
   
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
        videos= self._get_videos(video_list)
        video_clips=[]
        for vidx, video in enumerate(videos):
            clips = self._get_clips(video, size, step)
            for clip in clips:
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
            return [video_t]
        video_t = video_t.unfold(dim, size, step) # N HWC T
        video_t = video_t.permute(0, 4, 1, 2, 3)  # N T HWC
        return video_t
    
    def __getitem__(self, index):
        clip, vidx = self.video_clips[index]
        clip = self._to_pil_image(clip)
        
        if self.temporal_transform is not None:
            clip = self.temporal_transform(clip)
        
        if self.spatial_transform is not None:
            self.spatial_transform.randomize_parameters()
            clip = [self.spatial_transform(img) for img in clip]

        target = self.samples[1][vidx]
        clip = torch.stack(clip).transpose(0, 1) # TCHW-->CTHW
        target = torch.tensor(target).unsqueeze(0) # () -> (1,)
        return clip, target
    
    def _to_pil_image(self, video):
        video = [v.permute(2, 0, 1) for v in video] # for to_pil_image
        return [F.to_pil_image(img) for img in video]
    
    def __len__(self):
        return len(self.video_clips)
    
    #def __getpath__(self, index):
    #    return self.samples[0][index]
