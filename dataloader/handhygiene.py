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
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
from glob import glob
import logging


def make_dataset(dir, extensions=None, is_valid_file=None):
    folders=[]
    
    for fname in os.listdir(os.path.join(dir)):
        item = os.path.join(dir, fname)
        folders.append(item)
    return folders

def get_classes(label_txt):
    with open(label_txt) as f:
        actions = f.readlines()
    return [a.strip('\n').strip(' ') for a in actions]

    
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
        self.classes = get_classes('./data/annotations/hh_action_list.txt')
        self.class_to_idx = {self.classes[i]: i for i in range(len(self.classes))}
        
        if opt_flow_preprocess:
            self.preprocess(extensions[0], useCuda=False)
        
        annotation_path = './data/annotations/hh_target.csv'
        self.samples = make_dataset(self.root)
        
        self.frames_per_clip = frames_per_clip
        self.steps = self._init_steps(step_between_clips)
        self.video_list = self.samples
        
        # TODO: use video_utils subset
        self.optflow_list = [self._optflow_path(x) for x in self.video_list]
        self.video_clips = VideoClips(self.video_list, 
                                      frames_per_clip, 
                                      step_between_clips, 
                                      frame_rate,
                                      with_detection=with_detection,
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
        print("Number of clips: ", self._num_clips_per_class())
        print("Number of frames: ", self._num_frames_per_class())
        print("Total number of frames:", self._num_frames_in_clips_per_class())
        
    def _make_item(self, idx): 
        video, _, info, video_idx = self.video_clips.get_clip(idx)
        optflow, _, _, _ = self.optflow_clips.get_clip(idx)
        video = self._to_pil_image(video)
        optflow = self._to_pil_image(optflow)
        
        label = info['label']
        ## TODO:
        if label == 'removing_gloves':
            label = 'wearing_gloves'
        label = self.class_to_idx[label] # self.samples[video_idx][1] 
        
        '''
        keypoints = info['keypoints']
        if keypoints is None:
            print("idx: {} not having keypoints".format(idx))
        elif isinstance(keypoints, list) and len(keypoints) == 0:
            print("idx: {} not having keypoints".format(idx))
            
        rois = self._get_clip_coord(idx, keypoints)
        '''
        bboxes = self._get_bounding_box(info, align=False, padding=False)
        return (video, optflow, bboxes, label)
    
    def __getitem__(self, idx):
        video, optflow, rois, label = self._make_item(idx)
        
        if self.temporal_transform is not None:
            self.temporal_transform.randomize_parameters()
            video = self.temporal_transform(video)
            optflow = self.temporal_transform(optflow)
            rois = self.temporal_transform(rois)
            
        _, _, info, video_idx = self.video_clips.get_clip(idx)
        assert len(rois) == 16, print(self.video_list[video_idx], rois)

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
    
    def _init_steps(self, step_between_clips):
        if isinstance(step_between_clips, dict):
            self.steps = step_between_clips
        else:
            self.steps = {label:step_between_clips 
                          for label in self.classes}
        return self.steps

    
    def _num_clips_per_class(self):
        classes = self.classes
        num_clips = {cls:0 for cls in classes}
        
        for i in range(self.video_clips.num_clips()):
            (vidx, idx, pidx, cidx, label), vidx_c = self.video_clips.get_clip_location(i)
            # TODO:
            if label == 'removing_gloves':
                label = 'wearing_gloves'
            num_clips[label] += 1
        return num_clips
    
    def _num_frames_per_class(self):
        import pandas as pd
        classes = self.classes
        num_frames = {cls:0 for cls in classes}
        for i, v in enumerate(self.video_clips.video_pts):

            df = pd.DataFrame([row for row in self.video_clips.info if row[0] == i])
            rows = df[[0,1,4]].drop_duplicates().values
            for row in rows:
                label = row[-1]
                # TODO:
                if label == 'removing_gloves':
                    label = 'wearing_gloves'
                num_frames[label] += len(v)
        return num_frames
    
    def _num_frames_in_clips_per_class(self):
        import pandas as pd
        classes = self.classes
        num_frames = {cls:0 for cls in classes}
        for i, v in enumerate(self.video_clips.video_pts):

            df = pd.DataFrame([row for row in self.video_clips.info if row[0] == i])
            rows = df[[0,1,3,4]].drop_duplicates().values
            for row in rows:
                label = row[-1]
                # TODO:
                if label == 'removing_gloves':
                    label = 'wearing_gloves'
                num_frames[label] += len(v)
        return num_frames
    
    '''
    def _get_clip_coord(self, idx, coords):
        fpc = self.frames_per_clip
        vidx, cidx = self.video_clips.get_clip_location(idx)
        label = self.samples[vidx][1]
        step = self.steps[self.classes[label]]
        start= cidx * step
        rois = self._smoothing(coords)
        rois = rois[start:start+fpc]
        return rois
    '''
    '''
    def _get_bounding_box(self, info, align=True, padding=True): #, label):
        
        person_id = info['target_ids']
        bboxes = []
        for i in info['annotations']:
            #for person_id in person_ids:
            for shape in i['shapes']:
                if shape['group_id'] == int(person_id):
                    bbox = shape['points']

                    if bbox == [[0.0, 0.0]]:
                        bbox = [[0.0, 0.0], [0.0, 0.0]]
                    assert np.asarray(bbox).shape == (2,2), i['imagePath']

                    x1, y1, x2, y2 = np.reshape(np.asarray(bbox), (4))
                    w = x2 - x1
                    h = y2 - y1
                    bboxes.append([x1, y1, w, h])

            continue # dealing with a single target  
                
        if align:
            bboxes = [self.align_boundingbox(b) for b in bboxes]
        if padding:
            bboxes = self._padding(bboxes)
        
        return bboxes
    '''
    
    def _get_bounding_box(self, info, align=False, padding=False): #, label):
        
        person_id = info['target_id']
        bboxes = []
        for i in info['annotations']:
            bbox, _ = i[person_id]
            x1, y1, x2, y2 = bbox
            w = x2 - x1
            h = y2 - y1
            bboxes.append([x1, y1, w, h])

        if align:
            bboxes = [self.align_boundingbox(b) for b in bboxes]
        if padding:
            bboxes = self._padding(bboxes)
        
        return bboxes
    
    def align_boundingbox(self, roi):
        x, y, w, h = roi
        ratio = w/h
        if ratio > 1:
            y -= int((w-h)/2)
        else:
            x -= int((h-w)/2)
        return [x, y, w, w]

    def _padding(self, bboxes):
        rois_tmp = bboxes.copy()
        roi_empty = [0.0, 0.0, 0.0, 0.0]
        
        for i, roi in enumerate(rois_tmp):
            if roi == roi_empty:
                # find a closest roi
                bboxes[i] = [roi for roi in bboxes[i:] if roi != roi_empty][0]
        return bboxes
    
    '''
    def _smoothing(self, coords):
        
        from .poseroi import get_windows
        from .poseroi import calc_roi
        
        windows = get_windows(coords)
        if len(windows)==0:
            print("empty windows", windows)
            return windows
        
        rois = calc_roi(windows)
        
        return self._padding(rois)
    '''
    
    def preprocess(self, ext, useCuda=True):
        import sys
        sys.path.append('./utils/python-opencv-cuda/python')
        import common as cm

        phase = self.root
        for v_name in os.listdir(phase):
            
            ## TODO
            #if ext not in v_name:
            #    continue

            v_path = os.path.join(phase, v_name)
            v_output = self._optflow_path(v_path)
            flows = cm.findOpticalFlow(v_path, v_output, useCuda, True)

            ## TODO: write as a video
            #if flows is not None:
            #    flows = np.asarray(flows)
            #    write_video(v_output, flows, fps=15)
                    
    
    def _optflow_path(self, video_path):
        f_type = self._optflow_type()
        f_output = os.path.join(video_path, f_type)
        return f_output
    
    def _optflow_type(self):
        ### TODO: implement reversed version
        #'reverse_flow' if reversed else 'flow'
        return 'flow'
    
    def _get_clip_loc(self, idx):
        vidx, cidx = self.video_clips.get_clip_location(idx)
        vname, label = self.samples[vidx]
        return (vidx, cidx)

    def _to_pil_image(self, video):
        video = [v.permute(2, 0, 1) for v in video] # for to_pil_image
        return [F.to_pil_image(img) for img in video]
        
    def __len__(self):
        return self.video_clips.num_clips()
