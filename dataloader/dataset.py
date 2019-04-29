import torch
import torch.utils.data as data
import torch.nn.functional as F

import os
import numpy as np
from PIL import Image # to use torchivision transforms

from tqdm import tqdm
from glob import glob

import json
import pandas as pd
from .imagedataset import *
from .opticalflow import compute_TVL1
from .opticalflow import get_flow
from .poseroi import calc_margin
from .poseroi import crop_by_clip

from multiprocessing import Pool


def default_loader(dir):
    rgbs = pil_frame_loader(dir)
    flows = pil_flow_loader(dir)
    return rgbs, flows
            
    
def pil_frame_loader(path):
    """
        return: list of PIL Images
    """
    frames = sorted([os.path.join(path, img) for img in os.listdir(path)])
    
    buffer = []
    for i, fname in enumerate(frames):
        if not is_image_file(fname):
            continue
        with open(fname, 'rb') as f:
            img = Image.open(f)
            img = img.convert('RGB')
            buffer.append(img)   
    return buffer

def pil_flow_loader(dir):
    """
        return: list of PIL Images
    """
    flow = get_flow(dir)
   
    buffer = []
    for i, flw in enumerate(flow):
        shape = flw.shape
        # to make extra 3 channel to use torchvision transform
        tmp = np.empty((shape[0], shape[1], 1)).astype(np.uint8) 
        img = np.dstack((flw.astype(np.uint8), tmp))
        img = Image.fromarray(img)
        buffer.append(img)
    return buffer

class VideoDataset(data.Dataset):
        
    def __init__(self, root, split='train', clip_len=16, transform=None, preprocess=False,
                 loader=default_loader, num_workers=1):

        self.root = root
        self.loader = loader
        self.video_dir = os.path.join(root, 'videos')
        self.image_dir = os.path.join(root, 'images')
        folder = os.path.join(self.image_dir, split)
        
        classes, class_to_idx = find_classes(folder)
        self.classes = classes
        self.samples = make_dataset(folder, class_to_idx) # [fnames, labels]
        self.transform = transform
        self.clip_len = clip_len
        self.num_workers = num_workers

        if preprocess:
            self.preprocess(num_workers)

    def __getitem__(self, index):
        # loading and preprocessing.
        """
            TODO: clean up batch size ordering
            sampling 16 frames
        """
        fnames, targets = self.samples[0][index], self.samples[1][index]
        frames, flows = self.loader(fnames)
        
        if self.transform: ## applying torchvision transform
            _frames = []
            _flows = []
            for frame in frames: ## for RGB
                frame = self.transform(frame)
                _frames.append(frame)
            for flow in flows:
                ### TODO: flow should be from [-20, 20] to [-1, 1], now : [0, 255] to [-1, 1]
                flow = self.transform(flow) 
                _flows.append(flow[:-1,:,:]) # exclude temp channel 3
            frames = _frames
            flows = _flows
       
        # target transform
        #if len(targets) != 2:
        targets = torch.tensor(targets).unsqueeze(0)
        
        ## temporal transform
        frames, flows = self.temporal_transform((frames, flows), index)
        return frames, flows, targets
    
    def to_one_hot(self, label):
        to_one_hot = np.eye(2)
        return to_one_hot[int(label)]
    
    def temporal_transform(self, streams, index):
        """
            all clip length resize to 16
        """
        frames, flows = streams
        clip_len = self.clip_len
        nframes = len(frames)
        
        if nframes == clip_len:
            frames = torch.stack(frames).transpose(0, 1) # DCHW -> CDHW
            flows = torch.stack(flows).transpose(0, 1)
            #print(frames.shape, flows.shape, index)
            return (frames, flows)
        #elif nframes < clip_len:
        #    return self.clip_looping(streams, index)
        
        elif nframes > clip_len:
            return self.clip_sampling(streams, index)
        
        else:
            #frames = self.clip_speed_changer(frames)
            #flows = self.clip_speed_changer(flows)
            frames = torch.stack(frames).transpose(0, 1) # DCHW -> CDHW
            flows = torch.stack(flows).transpose(0, 1)
            return (frames, flows)
    
    def clip_looping(self, streams, index):
        """
            Loop a clip as many times as necessary to satisfy input size
            input shape: DxCxHxW
            return shape: CxDxHxW
        """
        frames, flows = streams
        clip_len = self.clip_len
        nframes = len(frames)
        
        niters = int(clip_len/nframes)+1
        frames = frames*niters
        frames = frames[:clip_len]
        flows = flows*niters
        flows = flows[:clip_len]
        frames = torch.stack(frames).transpose(0, 1)
        flows = torch.stack(flows).transpose(0, 1)
        
        return (frames, flows)
        
    def clip_speed_changer(self, images):
        """
            Interpolate clip size with length of `self.clip_len`
            ex) 25 --> 16
            input shape: DxCxHxW
            return shape: CxDxHxW
        """
        ## TODO: tensor ordering
        images = torch.stack(images)
        shape = images.shape # DCHW
        images = images.transpose(0, 1).unsqueeze(0) #DCHW --> BCDHW
        #`mini-batch x channels x [optional depth] x [optional height] x width`.
        images = F.interpolate(images, size=(self.clip_len, 224, 224), 
                               mode='trilinear', align_corners=True)
        images = images.view(shape[1], self.clip_len, shape[2], shape[3]) #squeeze
        return images #CDHW
    
    def clip_sampling(self, streams, index):
        """
            Sample clip with random start number.
            The length of sampled clip should be same with `self.clip_len`
            input shape: DxCxHxW
            return shape: CxDxHxW
        """
        frames, flows = streams
        clip_len = self.clip_len
        start = 0
        if len(frames) != len(flows):
            print(self.__getpath__(index))
            return print("number of frames {} and flows {} are different.".format(len(frames), len(flows)))
        
        nframes = len(frames)
        if nframes > clip_len:
            size = nframes-clip_len+1
            start = np.random.choice(size, 1)[0]
        elif nframes < clip_len: # drop a clip when its length is less than 16
            print(self.__getpath__(index))
            return print("minimum {} frames are needed to process".format(clip_len))
        
        frames = torch.stack(frames[start:start+clip_len]).transpose(0, 1)
        flows = torch.stack(flows[start:start+clip_len]).transpose(0, 1)
        return (frames, flows)
    
    def clip_add_backwards(self, streams, index):
        """ TODO"""
        return
    
    def preprocess(self, num_workers):
        paths = [self.__getpath__(i) for i in range(self.__len__())]
        pool = Pool(num_workers)
        pool.map(get_flow, paths)
        return

    
    def __len__(self):
        return len(self.samples[0]) # fnames
    
    def __getpath__(self, index):
        return self.samples[0][index]


if __name__ == "__main__":
    from torch.utils.data import DataLoader
    
    dataset_path = os.getcwd()
    train_dataset = VideoDataset(dataset_path, split='train', transform=transforms)
    train_loader = DataLoader(dataset['train'], batch_size=2, shuffle=True, num_workers=4)

    for i, sample in enumerate(train_loader):
        rgbs = sample[0]
        flows = sample[1]
        labels = sample[2]
        print(rgbs.size())
        print(flows.size())
        print(labels)

        if i == 1:
            break