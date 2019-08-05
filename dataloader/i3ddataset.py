import torch
import torch.utils.data as data
from torchvision.datasets.utils import list_dir
from torchvision.io.video import write_video
from .videodataset import VideoDataset
from .videodataset import make_dataset
from .video_utils import VideoClips

import os
import sys
sys.path.append('./utils/python-opencv-cuda/python')
import common as cm
import numpy as np

class I3DDataset(VideoDataset):
    def __init__(self, root, frames_per_clip, 
                 step_between_clips=1,
                 frame_rate=None,
                 spatial_transform=None,
                 temporal_transform=None,
                 opt_flow_preprocess=False):
        super(I3DDataset, self).__init__(root, frames_per_clip,
                                         step_between_clips=step_between_clips,
                                         frame_rate=frame_rate,
                                         spatial_transform=spatial_transform,
                                         temporal_transform=temporal_transform)
        extensions = ('',)
        classes = list(sorted(list_dir(root)))
        class_to_idx = {classes[i]: i for i in range(len(classes))}
        
        if opt_flow_preprocess:
            self.preprocess(extensions[0])
            
        self.samples = make_dataset(self.root, class_to_idx, extensions, is_valid_file=None)
        self.classes = classes
        self.video_list = [x[0] for x in self.samples if 'flow' not in x[0]]
        self.optflow_list = [x[0] for x in self.samples if '/flow' in x[0]]
        self.video_clips = VideoClips(self.video_list, frames_per_clip, step_between_clips, frame_rate)
        self.optflow_clips = VideoClips(self.optflow_list, frames_per_clip, step_between_clips, frame_rate)
        self.spatial_transform = spatial_transform
        self.temporal_transform = temporal_transform
    
    
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
                    write_video(v_output, flows, fps=15)