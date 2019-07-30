import torch
import torch.utils.data as data
from torchvision.datasets import DatasetFolder
from torchvision.datasets.folder import find_classes
from torchvision.datasets.folder import IMG_EXTENSIONS
from torchvision.datasets.folder import has_file_allowed_extension

import os
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
        return: list of PIL Images
    """
    video = []
    for i, fname in enumerate(frames):
        with open(fname, 'rb') as f:
            img = Image.open(f)
            img = img.convert('RGB')
            video.append(img)   
    return video


class VideoFolder(DatasetFolder):
    """A generic data loader where the samples are arranged in this way: ::

        root/class_x/video_name/images0001.ext
        root/class_x/video_name/images0030.ext
        root/class_x/xxz/images0001.ext

        root/class_y/123/images0001.ext
        root/class_y/nsdf3/images0001.ext
        root/class_y/asd932_/images0001.ext
    """
    def __init__(self, root, split='train', clip_len=16, 
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
        
        classes, class_to_idx = find_classes(folder)
        self.classes = classes
        self.class_to_idx = class_to_idx
        self.samples = make_dataset(folder, class_to_idx)
        self.spatial_transform = spatial_transform
        self.temporal_transform = temporal_transform
        self.target_transform = target_transform
        self.clip_len = clip_len
        
    def __getitem__(self, index):
        fnames = self.samples[0][index]
        findices = get_framepaths(fnames)
        
        if self.temporal_transform is not None:
            findices = self.temporal_transform(findices)
        clips = self.loader(findices)
        
        if self.spatial_transform is not None:
            clips = [self.spatial_transform(img) for img in clips]
        clips = torch.stack(clips).permute(1, 0, 2, 3)

        targets = self.samples[1][index]
        if self.target_transform is not None:
            target = self.target_transform(target)
            
        #targets = torch.tensor(targets).unsqueeze(0)
        return clips, targets
    
    
    def __len__(self):
        return len(self.samples[0]) # fnames
    
    def __getpath__(self, index):
        return self.samples[0][index]
