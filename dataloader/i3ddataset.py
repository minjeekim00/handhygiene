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
        
        if opt_flow_preprocess:
            self.preprocess()
            
        self.samples = make_dataset(self.root, class_to_idx, extensions, is_valid_file=None)
        self.classes = classes
        video_list = [x[0] for x in self.samples if 'flow' not in x[0]]
        optflow_list = [x[0] for x in self.samples if 'flow' in x[0]]
        self.video_clips = VideoClips(video_list, frames_per_clip, step_between_clips)
        self.optflow_clips = VideoClips(optflow_list, frames_per_clip, step_between_clips)
        self.spatial_transform = spatial_transform
        self.temporal_transform = temporal_transform
    
    
    def _optflow_path(self, video_path):
        f_type = self._optflow_type()
        v_name, v_ext = os.path.splitext(video_path)
        v_dir, v_name = os.path.split(v_name)
        
        if f_type in v_dir: # for flow dirs
            v_dir = v_dir.replace(f_type+'/') # flow/ 제거
            f_dir = os.path.join(v_dir, v_name, f_type)
        else:
            f_dir = os.path.join(v_dir, f_type)
            
        if not os.path.exists(f_dir):
            os.mkdir(f_dir)
            
        return os.path.join(f_dir, '{}{}'.format(v_name, v_ext))
    
    def _optflow_type(reversed=False):
        ### TODO: implement reversed version
        return 'reverse_flow' if reversed else 'flow'
    
    def __getitem__(self, idx):
        video, _, _, video_idx = self.video_clips.get_clip(idx)
        optflow, _, _, _ = self.optflow_clips.get_clip(idx)
        
        video = self._to_pil_image(video)
        optflow = self._to_pil_image(optflow)
        label = self.samples[video_idx][1]
        
        if self.temporal_transform is not None:
            video = [self.temporal_transform(img) for img in video]
            optflow = [self.temporal_transform(img) for img in optflow]
        
        if self.spatial_transform is not None:
            self.spatial_transform.randomize_parameters()
            video = [self.spatial_transform(img) for img in video]
            optflow = [self.spatial_transform(img) for img in optflow]
            
        video = torch.stack(video).transpose(0, 1) # TCHW-->CTHW
        optflow = torch.stack(optflow).transpose(0, 1) # TCHW-->CTHW
        label = torch.tensor(label).unsqueeze(0) # () -> (1,)
        return video, optflow, label
    
    def preprocess(self, useCuda=True):
        root = self.root
        i_path = root.replace('videos','images')
        
        for label in os.listdir(i_path):
            for i_name in os.listdir(os.path.join(i_path, label)):
                path = os.path.join(i_path, label, i_name)
                cm.findOpticalFlow(path, useCuda, True)
            
            ### TODO: make video
            
            #for phase in ['train', 'val', 'test']:
            #    for label in ['clean', 'notclean']:
            #        VIDEO_DIR = './data/videos/simulate/{}/{}/'.format(phase, label)
            #        IMAGE_DIR = './data/images/simulate/{}/{}/'.format(phase, label)

            #        for vname in os.listdir(IMAGE_DIR):
            #            fps = 15
            #            start = int(vname[-6:])
            #            i_path_flow = os.path.join(IMAGE_DIR, vname, 'flow', vname[:-6]+'%06d_flow.jpg')
            #            dst = os.path.join(VIDEO_DIR, 'flow', vname+'.mp4')
            #            if not os.path.exists(dst):
            #                !avconv -r $fps -start_number $start -i $i_path_flow -c:v libx264 -profile:v high -crf 20 -pix_fmt yuv420p $dst -y
            #if not hasVideo:
            #    fps = 15
            #    start = int(v_path[-6:])
            #    v_path_flow = self._optflow_path(v_name)
            #    i_path_flow = os.path.join(i_path, name, os.path.basename(v_path)[:-6]+'%06d_{}.jpg'.format(name))
            #    print('avconv -r {} -start_number {} -i {} -c:v libx264 -profile:v high -crf 20 -pix_fmt yuv420p {} -y'.format(fps, start, i_path_flow, v_path_flow))
            #    os.system('avconv -r {} -start_number {} -i {} -c:v libx264 -profile:v high -crf 20 -pix_fmt yuv420p {} -y'.format(fps, start, i_path_flow, v_path_flow))
            