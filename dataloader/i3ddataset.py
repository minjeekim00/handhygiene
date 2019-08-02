import torch
import torch.utils.data as data
from torchvision.datasets.video_utils import VideoClips
from torchvision.datasets.utils import list_dir
from torchvision.datasets.folder import make_dataset
from .videodataset import VideoDataset

import os
import sys
sys.path.append('./utils/python-opencv-cuda/python')
import common as cm



class I3DDataset(VideoDataset):
    def __init__(self, root, frames_per_clip, step_between_clips=1, 
                 spatial_transform=None,
                 temporal_transform=None,
                 opt_flow_preprocess=False):
        super(I3DDataset, self).__init__(root, frames_per_clip, 
                                         step_between_clips=step_between_clips,
                                         spatial_transform=spatial_transform,
                                         temporal_transform=temporal_transform)
        
        extensions = ('mp4',)
        classes = list(sorted(list_dir(root)))
        class_to_idx = {classes[i]: i for i in range(len(classes))}
        self.samples = make_dataset(self.root, class_to_idx, extensions, is_valid_file=None)
        self.classes = classes
        video_list = [x[0] for x in self.samples]
        optflow_list =  [self._optflow_path(x[0]) for x in self.samples]
                                      
        self.video_clips = VideoClips(video_list, frames_per_clip, step_between_clips)
        self.optflow_clips = VideoClips(optflow_list, frames_per_clip, step_between_clips)
        
        if opt_flow_preprocess:
            self.preprocess()
        #self.transform = transform
        self.spatial_transform = spatial_transform
        self.temporal_transform = temporal_transform
    
    
    def _optflow_path(self, video_path):
        v_name, v_ext = os.path.splitext(video_path)
        return '{}_flow{}'.format(v_name, v_ext)
                                      
    def __getitem__(self, index):
        # loading and preprocessing.
        fnames= self.samples[0][index]
        findices = get_framepaths(fnames)
        target = self.samples[1][index]
        
        if self.temporal_transform is not None:
            findices = self.temporal_transform(findices)
        clips, flows = self.loader(findices)
         
        if self.spatial_transform is not None:
            self.spatial_transform.randomize_parameters()
            clips = [self.spatial_transform(img) for img in clips]
            flows = [self.spatial_transform(img) for img in flows]
        
        if self.target_transform is not None:
            target = self.target_transform(target)
        target = torch.tensor(target).unsqueeze(0)
       
        clips = torch.stack(clips).permute(1, 0, 2, 3)
        flows = [flow[:-1,:,:] for flow in flows]
        flows = torch.stack(flows).permute(1, 0, 2, 3)
        
        return clips, flows, target
    
    def preprocess(self, useCuda=True):
        v_paths = self.samples
        i_paths = [p.replace('videos','images') for p in v_paths]
        i_paths = [os.path.splitext(p)[0] for p in i_paths]
        
        for path in tqdm(paths):
            cm.findOpticalFlow(path, useCuda, True, False, False)