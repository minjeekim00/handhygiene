import torch
import torch.utils.data as data
from torchvision.datasets.utils import list_dir
from torchvision.datasets.folder import IMG_EXTENSIONS
from torchvision.datasets.folder import has_file_allowed_extension
from torchvision.io.video import write_video
from torchvision.datasets.vision import VisionDataset
from torchvision.transforms import functional as F
from .video_utils import VideoClips

import os
import sys
sys.path.append('./utils/python-opencv-cuda/python')
import common as cm
import numpy as np
from PIL import Image
from tqdm import tqdm
from glob import glob
import logging


def make_dataset(dir, class_to_idx, extensions=None, is_valid_file=None):
    exclusions = ['40_20190208_frames026493',
                  '34_20190110_frames060785', #window
                  '34_20190110_frames066161',
                  '34_20190110_frames111213']
    folders=[]
    for label in os.listdir(os.path.join(dir)):
        for fname in os.listdir(os.path.join(dir, label)):
            
            # exceptions
            if any([fname in ex for ex in exclusions]):
                continue
            txtfile = os.path.join(dir, label, fname, fname+'.txt')
            if not os.path.exists(txtfile):
                print("{} not exists".format(txtfile))
                continue
            from .io.video import target_dataframe
            df = target_dataframe()
            hastarget = len(df[df['imgpath']==fname].values)
            if not hastarget:
                print("{} doesnt' have target".format(fname))
                continue
            
            # append item
            item = (os.path.join(dir, label, fname), class_to_idx[label])
            folders.append(item)
    return folders

    
class HandHygiene(VisionDataset):
    def __init__(self, root, frames_per_clip, 
                 step_between_clips=1,
                 frame_rate=None,
                 downsample_size=None,
                 openpose_transform=None,
                 spatial_transform=None,
                 temporal_transform=None,
                 opt_flow_preprocess=False, 
                 with_detection=True):

        super(HandHygiene, self).__init__(root)
        extensions = ('',) #tmp
        classes = list(sorted(list_dir(root)))[::-1]
        class_to_idx = {classes[i]: i for i in range(len(classes))}
        
        if opt_flow_preprocess:
            self.preprocess(extensions[0])
            
        self.samples = make_dataset(self.root, class_to_idx)
        self.classes = classes
        
        # TODO: use video_utils subset
        self.optflow_list = [self._optflow_path(x) for x in self.video_list]
        self.video_clips = VideoClips(self.video_list, 
                                      frames_per_clip, 
                                      step_between_clips, 
                                      frame_rate,
                                      with_detection=with_detection
                                      downsample_size=downsample_size)
        
        self.optflow_clips = VideoClips(self.optflow_list, 
                                        frames_per_clip, 
                                        step_between_clips, 
                                        frame_rate)
        
        self.spatial_transform = spatial_transform
        self.temporal_transform = temporal_transform
        self.openpose_transform = openpose_transform
        
        print('Number of {} video clips: {:d} ({:d} images)'.format(
            root, self.video_clips.num_clips(), self.video_clips.num_total_frames()))
        print("Number of clips per class: ", self._num_clips_per_class())
        print("Number of frames per class: ", self._num_frames_per_class())
        
        
    def _make_item(self, idx): 
        video, _, info, video_idx = self.video_clips.get_clip(idx)
        optflow, _, _, _ = self.optflow_clips.get_clip(idx)
        video = self._to_pil_image(video)
        optflow = self._to_pil_image(optflow)
        label = self.samples[video_idx][1]
        keypoints = info['body_keypoint']
        if keypoints is None:
            print("idx: {} not having keypoints".format(idx))
        elif isinstance(keypoints, list) and len(keypoints) == 0:
            print("idx: {} not having keypoints".format(idx))
            
        rois = self._get_clip_coord(idx, keypoints)
        return (video, optflow, rois, label)
    
    def __getitem__(self, idx):
        video, optflow, rois, label = self._make_item(idx)
        
        if self.temporal_transform is not None:
            self.temporal_transform.randomize_parameters()
            video = self.temporal_transform(video)
            optflow = self.temporal_transform(optflow)
            rois = self.temporal_transform(rois)
            
        if self.openpose_transform is not None:
            self.openpose_transform.randomize_parameters()
            video = [self.openpose_transform(v, rois, i) 
                     for i, v in enumerate(video)]
            optflow = [self.openpose_transform(f, rois, i) 
                       for i, f in enumerate(optflow)]
            if len(video)==0:
                print("windows empty")
                
        if self.spatial_transform is not None:
            self.spatial_transform.randomize_parameters()
            video = [self.spatial_transform(img) for img in video]
            optflow = [self.spatial_transform(img) for img in optflow]
            
        video = torch.stack(video).transpose(0, 1) # TCHW-->CTHW
        optflow = torch.stack(optflow).transpose(0, 1) # TCHW-->CTHW
        optflow = optflow[:-1,:,:,:] # 3->2 channel
        label = torch.tensor(label)
        
        if len(self.classes) < 2:
            label = label.unsqueeze(0) # () -> (1,) for binary classification
            
        return video, optflow, label
    
    def _get_clip_coord(self, idx, coords):
        fpc = self.frames_per_clip
        vidx, cidx = self.video_clips.get_clip_location(idx)
        target = self.samples[vidx][1]
        step = self.steps[self.classes[target]]
        start= cidx * step
        rois = self._smoothing(coords)
        rois = rois[start:start+fpc]
        return rois
    
    def _smoothing(self, coords):
        from .poseroi import get_windows
        from .poseroi import calc_roi
        
        windows = get_windows(coords)
        if len(windows)==0:
            print("empty windows", windows)
            return windows
        
        rois = calc_roi(windows)
        for i, roi in enumerate(rois):
            if roi==None:
                rois[i] = [roi for roi in rois_tmp if roi != None][0]
        return rois
    
    def preprocess(self, ext, useCuda=True):
        root = self.root
        for label in os.listdir(root):
            for v_name in os.listdir(os.path.join(root, label)):
                ## TODO
#                 if ext not in v_name:
#                     continue
                v_path = os.path.join(root, label, v_name)
                v_output = self._optflow_path(v_path)
                flows = cm.findOpticalFlow(v_path, v_output, useCuda, True)
            
                ## TODO: write as a video
#                 if flows is not None:
#                     flows = np.asarray(flows)
#                     write_video(v_output, flows, fps=15)
                    
    
    def _optflow_path(self, video_path):
        f_type = self._optflow_type()
        f_output=os.path.join(video_path, f_type)
        return f_output
    
    def _optflow_type(self):
        ### TODO: implement reversed version
        #'reverse_flow' if reversed else 'flow'
        return 'flow'
    