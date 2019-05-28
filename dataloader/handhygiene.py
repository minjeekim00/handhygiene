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
            
            
    def __getitem__(self, index, augnum=None):
        # loading and preprocessing.
        fnames= self.samples[0][index]
        findices = get_framepaths(fnames)
        coords= self.samples[1][index]
        
        #if self.temporal_transform is not None:
        #    findices, coords = self.temporal_transform(findices, coords)
        clips, flows = self.loader(findices, coords)
        
        if self.openpose_transform is not None:
            self.openpose_transform.randomize_parameters()
            streams = [self.openpose_transform(img, flow, coords, i)
                       for i, (img, flow) in enumerate(zip(clips, flows))]
            clips = [stream[0] for stream in streams]
            flows = [stream[1] for stream in streams]
            
        if self.spatial_transform is not None:
            self.spatial_transform.randomize_parameters()
            clips = [self.spatial_transform(img) for img in clips]
            flows = [self.spatial_transform(img) for img in flows]
            
        target = self.samples[2][index]
        if self.target_transform is not None:
            target = self.target_transform(target)
        ### for augmentation
        clips, flows = self.save_augmentation(clips, flows, target, index, augnum)
            
        target = torch.tensor(target).unsqueeze(0)
        return clips, flows, target
        
        
        clips = torch.stack(clips).permute(1, 0, 2, 3)
        flows = [flow[:-1,:,:] for flow in flows]
        flows = torch.stack(flows).permute(1, 0, 2, 3)
        return clips, flows, target
        #return clips, flows, target, coords
    
    def save_augmentation(self, clips, flows, target, index, augnum):
        import matplotlib.pyplot as plt
        
        fnames = self.samples[0][index]
        datadir= os.path.split(fnames)[0]
        basename = os.path.split(fnames)[1]
        
        if not '_aug_' in basename:
            return clips, flows
        
        targetdir = self.get_augmented_name(fnames, target, augnum)
        
        imgs_orig = sorted(glob(os.path.join(fnames, '*.jpg')))
        imgs_flows = sorted(glob(os.path.join(fnames, 'flow', '*.jpg')))
        assert len(imgs_orig) == len(imgs_flows)
        
        for i in range(len(clips)):
            # writing augmented images
            # RGB
            img = imgs_orig[i]
            rgbarray = np.array(clips[i])
            rgbpath =self.get_target_path(targetdir, img)
            plt.imsave(rgbpath, rgbarray)
            
            # FLOW
            img = imgs_flows[i]
            #img_aug = os.path.basename(self.get_augmented_name(img, target))
            flowarray = np.array(flows[i])
            flowdir = os.path.join(targetdir, 'flow')
            if not os.path.exists(flowdir):
                os.mkdir(flowdir)
            flowpath = self.get_target_path(flowdir, img, True)
            plt.imsave(flowpath, flowarray)
        return clips, flows

    
    def get_augmented_name(self, fnames, target, augnum):
        str_aug = '_'
        datadir = os.path.splitext(fnames)[0]
        
        #num = len([path for path in glob(fnames+'*') if str_aug in path])
        #if str_aug == '_':
        #    num -= 1
        
        if target == 1:
            datadir = datadir.replace('clean', 'notclean')
        
        dirname = datadir #+str_aug+str(num)
        dirname += '_aug_'+str(augnum)
        if not os.path.exists(dirname):
            os.mkdir(dirname)
            print("target directory: {}".format(dirname))
        return dirname
    
    def get_target_path(self, targetdir, img, isFlowDir=False):
        
        if isFlowDir:
            targetdir = os.path.split(targetdir)[0]
            
        datadir = os.path.split(targetdir)[0]
        tbasename = os.path.split(targetdir)[1]
        
        imgdir = os.path.split(img)[0]
        imgname = os.path.split(img)[1]
        ibasename = os.path.basename(imgdir)
        if isFlowDir:
            ibasename = os.path.basename(os.path.split(imgdir)[0])
        
        added = tbasename.replace(ibasename, '')
        timgname = os.path.splitext(imgname)[0]+added+os.path.splitext(imgname)[1]
        
        if isFlowDir:
            return os.path.join(targetdir, 'flow', timgname)
        
        return os.path.join(targetdir, timgname)