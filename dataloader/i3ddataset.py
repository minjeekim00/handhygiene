import torch
import torch.utils.data as data
from torchvision.datasets.utils import list_dir
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

def make_dataset(dir, class_to_idx, extensions=None, is_valid_file=None):
    import os
    folders = []
    dir = os.path.expanduser(dir)
    if not ((extensions is None) ^ (is_valid_file is None)):
        raise ValueError("Both extensions and is_valid_file cannot be None or not None at the same time")
    if extensions is not None:
        def is_valid_file(x):
            return has_file_allowed_extension(x, extensions)
    for target in sorted(class_to_idx.keys()):
        d = os.path.join(dir, target)
        if not os.path.isdir(d):
            continue
        for root, _, fnames in sorted(os.walk(d)):
            if len(fnames) == 0:
                continue
            if all([is_valid_file(os.path.join(root, fname)) for fname in fnames]):
                item = (root, class_to_idx[target])
                folders.append(item)
    return folders

class I3DDataset(VisionDataset):
    def __init__(self, root, frames_per_clip, 
                 step_between_clips=1,
                 frame_rate=None,
                 downsample=None,
                 spatial_transform=None,
                 temporal_transform=None,
                 opt_flow_preprocess=False):
        
        super(I3DDataset, self).__init__(root)
        extensions = ('',)
        classes = list(sorted(list_dir(root)))
        class_to_idx = {classes[i]: i for i in range(len(classes))}
        
        if opt_flow_preprocess:
            self.preprocess(extensions[0])
            
        self.samples = make_dataset(self.root, class_to_idx, extensions, is_valid_file=None)
        self.classes = classes
        
        # TODO: use video_utils subset
        self.optflow_list = [os.path.join(x, 'flow') for x in self.video_list]
        
        self.video_clips = VideoClips(self.video_list, 
                                      frames_per_clip, 
                                      step_between_clips, 
                                      frame_rate,
                                      downsample_size=downsample)
        
        self.optflow_clips = VideoClips(self.optflow_list, 
                                        frames_per_clip, 
                                        step_between_clips, 
                                        frame_rate)
        
        self.spatial_transform = spatial_transform
        self.temporal_transform = temporal_transform
        if opt_flow_preprocess:
            self.preprocess(extensions[0])
    
    def _optflow_path(self, video_path):
        f_type = self._optflow_type()
        v_dir, v_name = os.path.split(video_path)
        f_dir = os.path.join(v_dir, f_type)
            
        if not os.path.exists(f_dir):
            print("creating flow directory: {}".format(f_dir))
            os.mkdir(f_dir)
        f_output=os.path.join(f_dir, v_name)
        return f_output
    
    def _optflow_type(self):
        ### TODO: implement reversed version
        #'reverse_flow' if reversed else 'flow'
        return 'flow'
    
    def __getitem__(self, idx):
        video, _, _, video_idx = self.video_clips.get_clip(idx)
        optflow, _, _, _ = self.optflow_clips.get_clip(idx)
        
        video = self._to_pil_image(video)
        optflow = self._to_pil_image(optflow)
        label = self.samples[video_idx][1]
        
        if self.temporal_transform is not None:
            self.temporal_transform.randomize_parameters()
            video = self.temporal_transform(video)
            optflow = self.temporal_transform(optflow)
        
        if self.spatial_transform is not None:
            self.spatial_transform.randomize_parameters()
            video = [self.spatial_transform(img) for img in video]
            optflow = [self.spatial_transform(img) for img in optflow]
            
        video = torch.stack(video).transpose(0, 1) # TCHW-->CTHW
        optflow = torch.stack(optflow).transpose(0, 1) # TCHW-->CTHW
        optflow = optflow[:-1,:,:,:] # 3->2 channel
        label = torch.tensor(label).unsqueeze(0) # () -> (1,)
        return video, optflow, label
    
    def preprocess(self, ext, useCuda=True):
        root = self.root
        for label in os.listdir(root): 
            for v_name in os.listdir(os.path.join(root, label)):
                if ext not in v_name:
                    continue
                v_path = os.path.join(root, label, v_name)
                v_output = self._optflow_path(v_path)
                flows = cm.findOpticalFlow(v_path, v_output, useCuda, True)
                if flows is not None:
                    flows = np.asarray(flows)
                    #write_video(v_output, flows, fps=15)