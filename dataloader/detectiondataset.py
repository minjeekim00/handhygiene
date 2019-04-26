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
from .imageset import *
from .opticalflow import compute_TVL1
from .opticalflow import get_flow
from .poseroi import calc_margin
from .poseroi import crop_by_clip


def make_dataset(dir, class_to_idx):
    """
        fnames: name of directory containing images
        coords: dict containg people, torso coordinates
        labels: class
    """
    fnames, coords, labels = [], [], []
    
    keypoint = '/data/private/minjee-video/handhygiene/data/keypoints.txt'
    df=pd.read_csv('/data/private/minjee-video/handhygiene/data/label.csv')
    
    exclusions = [
    '38_20190119_frames000643'
    ]
    # target 있는 imgpath만 선별
    df = df[df['targets'].notnull()]
    lists = df['imgpath'].values
    
    with open(keypoint, 'r') as file:
        data = json.load(file)
        data = data['coord']
        for fname in os.listdir(os.path.join(dir)):
            if is_image_file(fname):
                continue
            if fname not in lists:
                continue
            if fname in exclusions:
                continue
                
            frames = sorted([os.path.join(dir, img) 
                             for img in os.listdir(os.path.join(dir, fname))])
            frames = [img for img in frames if is_image_file(img)]
            
            item = [d for d in data if d['imgpath']==fname][0]
            people = item['people']
            torso = item['torso']
            npeople = len(item['people'])
            
            tidxs = df[df['imgpath']==fname]['targets'].values[0] # target idx
            tidxs = [int(t) for t in tidxs.strip().split(',')]
            nidxs = list(range(npeople))
            nidxs = [int(n) for n in nidxs if n not in tidxs]
            
            ## appending clean
            for tidx in tidxs:
                if len(frames) != len(people[tidx]):
                    print("<{}> {} coords and {} frames / of people {}"
                          .format(fname, len(people[tidx]), len(frames), tidx))
                    print(people[tidx])
                    continue
                fnames.append(os.path.join(dir, fname))
                coords.append({'people':people[tidx], 
                               'torso':torso[tidx]})
                labels.append('clean')
        
            ## appending notclean 
            if len(nidxs) > 0:
                max = np.random.randint(1, 2+1)
                for nidx in nidxs[:max]:
                    if len(frames) != len(people[nidx]):
                        print("<{}> {} coords and {} frames / of people {}"
                              .format(fname, len(people[nidx]), len(frames), nidx))
                        print(people[nidx])
                        continue

                    fnames.append(os.path.join(dir, fname))
                    coords.append({'people':people[nidx], 
                                   'torso':torso[nidx]})
                    labels.append('notclean')
    
    #assert len(labels) == len(fnames)
    print('Number of {} people: {:d}'.format(dir, len(fnames)))
    targets = labels_to_idx(labels)
    
    return [fnames, coords, targets]

def default_loader(dir, coords):
    
    rgbs = pil_frame_loader(dir, coords)
    flows = pil_flow_loader(dir, coords)
    return rgbs, flows
 
def crop_pil_image(coords, idx):
    
    people = coords['people']
    torso = coords['torso']
    
    try:
        window = people[idx]
        if window is not None:
            window = calc_margin(torso, window)
        else:
            window = (0, 0, 0, 0)
    except:
        print("{} fail to calculate margin".format(idx))
        window = None
        
    return window


def pil_frame_loader(dir, coords):
    """
        return: list of PIL Images
    """
    frames = sorted([os.path.join(dir, img) for img in os.listdir(dir)])
    frames = [fname for fname in frames if is_image_file(fname)]
    buffer = []
    cropped = [] # coordinates
    for i, fname in enumerate(frames):
        # calc margin
        window = crop_pil_image(coords, i)
        cropped.append(window)
        with open(fname, 'rb') as f:
            img = Image.open(f)
            img = img.convert('RGB')
            buffer.append(img)
    
    buffer = crop_by_clip(buffer, cropped)
    return buffer

def pil_flow_loader(dir, coords):
    """
        return: list of PIL Images
    """
    flow = get_flow(dir)
   
    buffer = []
    cropped = [] # coordinates
    for i, flw in enumerate(flow):
        window = crop_pil_image(coords, i)
        cropped.append(window)
        
        shape = flw.shape
        # to make extra 3 channel to use torchvision transform
        tmp = np.empty((shape[0], shape[1], 1)).astype(np.uint8) 
        img = np.dstack((flw.astype(np.uint8), tmp))
        img = Image.fromarray(img)
        buffer.append(img)
    
    buffer = crop_by_clip(buffer, cropped, 'flow')
    return buffer


class VideoDataset(data.Dataset):
        
    def __init__(self, root, split='train', clip_len=16, transform=None, preprocess=False, 
                 use_keypoints=False, loader=default_loader, num_workers=1):

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
        self.use_keypoints = use_keypoints

        if preprocess:
            self.preprocess(num_workers)

    def __getitem__(self, index):
        # loading and preprocessing.
        """
            TODO: clean up batch size ordering
            sampling 16 frames
        """
        fnames = self.samples[0][index]
        coords = self.samples[1][index]
        targets = self.samples[2][index]
        
        frames, flows = self.loader(fnames, coords)
        
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
        
        #if nframes == clip_len:
        #    return (frames, flows)
        #elif nframes < clip_len:
        #    return self.clip_looping(streams, index)
        #else:
            #frames = self.clip_speed_changer(frames)
            #flows = self.clip_speed_changer(flows)
            
        if nframes > clip_len:
            (frames, flows) = self.clip_sampling(streams, index)
        
        elif nframes < clip_len:
            frames = self.clip_speed_changer(frames)
            flows = self.clip_speed_changer(flows)
            return (frames, flows)
        
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
        
        frames = frames[start:start+clip_len]
        flows = flows[start:start+clip_len]
        return (frames, flows)
    
    def clip_add_backwards(self, streams, index):
        return
    
    def check_preprocess(self):
        # TODO: Check image size in image_dir
        if not os.path.exists(self.image_dir):
            return False
        elif not os.path.exists(os.path.join(self.image_dir, 'train')):
            return False
        return True
    
    
    def preprocess(self, num_workers):
        paths = [self.__getpath__(i) for i in range(self.__len__())]
        pool = Pool(num_workers)
        pool.map(get_flow, paths)
        return

    
    def __len__(self):
        return len(self.samples[0]) # fnames
    
    def __getpath__(self, index):
        return self.samples[0][index]
    
    def __getcoords__(self, index):
        return self.samples[1][index]


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