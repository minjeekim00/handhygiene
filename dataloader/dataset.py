import torch
import torch.utils.data as data

import os
import numpy as np
from PIL import Image # to use torchivision transforms
import cv2 # to do flow preprocessing 

from glob import glob


def make_dataset(dir, class_to_idx):
    fnames, labels = [], []
    lists = sorted(os.listdir(dir))
    
    for label in sorted(os.listdir(dir)):
        for fname in os.listdir(os.path.join(dir, label)):
            if os.path.splitext(fname)[1] == '.npy':
                continue
            fnames.append(os.path.join(dir, label, fname))
            labels.append(label)
            
    assert len(labels) == len(fnames)
    print('Number of {} videos: {:d}'.format(dir, len(fnames)))
    targets = labels_to_idx(labels)
    
    return [fnames, targets]

def labels_to_idx(labels):
    
    labels_dict = {label: i for i, label in enumerate(sorted(set(labels)))}
    #if len(set(labels)) == 2:
    #    return np.array([np.eye(2)[int(labels_dict[label])] for label in labels])
    #else:
    return np.array([labels_dict[label] for label in labels], dtype=int)

def find_classes(dir):
    """
       returns classes, class_to_idx
    """
    classes = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
    classes.sort()
    class_to_idx = {classes[i]: i for i in range(len(classes))}
    return classes, class_to_idx

def pil_frame_loader(path):
    """
        return: list of PIL Images
    """
    frames = sorted([os.path.join(path, img) for img in os.listdir(path)])
    
    buffer = []
    for i, fname in enumerate(frames):
        if os.path.splitext(fname)[1] == '.npy':
            continue
        with open(fname, 'rb') as f:
            img = Image.open(f)
            img = img.convert('RGB')
            buffer.append(img)
            
    #buffer = np.asarray(buffer, dtype=np.float32)       
    return buffer

############################ for flow processing
    
def compute_TVL1(prev, curr, bound=15):
    """Compute the TV-L1 optical flow."""
    TVL1 = cv2.DualTVL1OpticalFlow_create()
    flow = TVL1.calc(prev, curr, None)
    assert flow.dtype == np.float32

    flow = (flow + bound) * (255.0 / (2*bound))
    flow = np.round(flow).astype(int)
    flow[flow >= 255] = 255
    flow[flow <= 0] = 0
    
    return flow
   
def get_flow(dir):
    
    basename = os.path.basename(dir)
    flow_dir = os.path.join(dir,'{}.npy'.format(basename))
    if os.path.exists(flow_dir):
        return np.load(flow_dir)
    
    print("processing optical flows..... this will take for a while....")
    frames = glob(os.path.join(dir, '*.jpg'))
    frames.sort()
    
    flow = []
    prev = cv2.imread(frames[0])
    prev = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
    for i, frame_curr in enumerate(frames):
        curr = cv2.imread(frame_curr)
        curr = cv2.cvtColor(curr, cv2.COLOR_BGR2GRAY)
        tmp_flow = compute_TVL1(prev, curr)
        flow.append(tmp_flow)
        prev = curr
    
    np.save(flow_dir, flow) 
    return flow


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

#########################################   


class VideoDataset(data.Dataset):
    #def __init__(self, root, transform=None, target_transform=None,
    #             loader=default_loader):
        
    def __init__(self, root, split='train', clip_len='16', transform=None, target_transform=None, preprocess=False, loader=pil_frame_loader):

        self.root = root
        self.loader = pil_frame_loader
        self.video_dir = os.path.join(root, 'videos')
        self.image_dir = os.path.join(root, 'images')
        folder = os.path.join(self.image_dir, split)
        
        classes, class_to_idx = find_classes(folder)
        samples = make_dataset(folder, class_to_idx) # [fnames, labels]
        self.samples = samples
        self.transform = transform
        self.target_transform = target_transform
        self.clip_len = clip_len

        if preprocess:
            self.preprocess()

    def __getitem__(self, index):
        # loading and preprocessing.
        fnames, targets = self.samples[0][index], self.samples[1][index]
        frames = self.loader(fnames)
        flows = pil_flow_loader(fnames)
        
        streams = (frames, flows)
        if self.transform: ## applying torchvision transform
            _frames = []
            _flows = []
            for frame in frames: ## for RGB
                frame = self.transform(frame)
                _frames.append(frame)
            for flow in flows:
                ### TODO: flow should be from [-20, 20] to [-1, 1]
                # but for now [0, 255] to [-1, 1] for easy implement
                flow = self.transform(flow) 
                _flows.append(flow[:,:,:-1]) # exclude temp channel 3
            frames = _frames
            flows = _flows
       
        if self.target_transform:
            targets = self.target_transform(targets)
            
        targets = torch.tensor(targets).unsqueeze(0)
        
        return frames, flows, targets

    
    def labels_to_idx(labels):
        return {label: i for i, label in enumerate(sorted(set(labels)))}
    
    def to_one_hot(label):
        to_one_hot = np.eye(2)
        return to_one_hot[int(label)]
        
    def check_preprocess(self):
        # TODO: Check image size in image_dir
        if not os.path.exists(self.image_dir):
            return False
        elif not os.path.exists(os.path.join(self.image_dir, 'train')):
            return False
        return True
    
    
    def preprocess(self):
        from sklearn.model_selection import train_test_split
        #### TODO: split train test 
        print('Preprocessing finished.')

    
    def __len__(self):
        return len(self.samples[0]) # fnames



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