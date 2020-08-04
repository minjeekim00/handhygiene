import torch
import torch.utils.data as data
from torch.utils.data.sampler import BatchSampler

from torchvision.datasets.utils import list_dir
from torchvision.datasets.folder import IMG_EXTENSIONS
from torchvision.datasets.folder import has_file_allowed_extension
from torchvision.io.video import write_video
from torchvision.datasets.vision import VisionDataset
from torchvision.transforms import functional as F
from .video_utils import VideoClips
from .handhygiene import HandHygiene
from .handhygiene import make_dataset
from .handhygiene import get_annotations
from .handhygiene import has_target_action
from .handhygiene import get_classes

import os
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
from glob import glob
import logging

    
class HandHygiene_Triplet(HandHygiene):
    def __init__(self, root, frames_per_clip, 
                 step_between_clips=1,
                 frame_rate=None,
                 downsample_size=None,
                 openpose_transform=None,
                 spatial_transform=None,
                 temporal_transform=None,
                 opt_flow_preprocess=False, 
                 with_detection=True, 
                 label_list_path='./data/annotations/hh_action_list.txt',
                 annotation_path='./data/annotations/hh_target.csv'):

        super(HandHygiene, self).__init__(root)
        extensions = ('',) #tmp
        #classes = list(sorted(list_dir(root)))[::-1]
        self.classes = get_classes(label_list_path)
        self.class_to_idx = {self.classes[i]: i for i in range(len(self.classes))}
        self.df_annotations  = pd.read_csv(annotation_path)
        
        ### TEMP (label change)
        df = self.df_annotations.copy()
        df.loc[df['action']=='removing_gloves', 'action'] = 'wearing_gloves'
        self.df_annotations = df
        
        if opt_flow_preprocess:
            self.preprocess(extensions[0], useCuda=False)
        
        #self.samples = make_dataset(self.root, class_to_idx, annotation_path)
        self.samples = make_dataset(self.root, self.classes, self.df_annotations)
        
        self.frames_per_clip = frames_per_clip
        self.steps = self._init_steps(step_between_clips)
        self.video_list = [x[0] for x in self.samples]
        
        # TODO: use video_utils subset
        self.optflow_list = [self._optflow_path(x) for x in self.video_list]
        
        self.video_clips = VideoClips(self.video_list, 
                                      frames_per_clip, 
                                      step_between_clips, 
                                      frame_rate,
                                      with_detection=with_detection,
                                      downsample_size=downsample_size,
                                      annotation=self.df_annotations,
                                      target_classes=self.classes)
        
        self.optflow_clips = VideoClips(self.optflow_list, 
                                        frames_per_clip, 
                                        step_between_clips, 
                                        frame_rate,
                                        annotation=self.df_annotations,
                                        target_classes=self.classes)
        
        self.spatial_transform = spatial_transform
        self.temporal_transform = temporal_transform
        self.openpose_transform = openpose_transform
        
        print('Number of {} video clips: {:d} ({:d} images)'.format(
            root, self.video_clips.num_clips(), self.video_clips.num_total_frames()))
        print("Number of clips: ", self._num_clips_per_class())
        print("Number of frames: ", self._num_frames_per_class())
        print("Total number of frames:", self._num_frames_in_clips_per_class())
        
        
        _vidxs = [s[0] for s in self.video_clips.info]
        self.labels = [s[-1] for s in self.video_clips.info]
        self.label_to_indices = {label: np.where(np.array(self.labels) == label)[0]
                                             for label in set(self.labels)}
        
        if not self._is_train: 
            # Creates fixed triplets for testing
            # https://github.com/adambielski/siamese-triplet/blob/master/datasets.py
            random_state = np.random.RandomState(42)
            triplets = [[i,
                         #random_state.choice(label_to_indices[self.labels[i]]),
                         random_state.choice(
                             [lidx for lidx in label_to_indices[self.labels[i]]
                             if _vidxs[lidx] != _vidxs[i]] # positive sample이 같은 비디오에서 나오지 않게
                         ),
                         random_state.choice(
                             [lidx for lidx in label_to_indices[
                                 np.random.choice(
                                     list(set(self.labels) - set([self.labels[i]])))
                             ] if _vidxs[lidx] != _vidxs[i]])]
                        for i in range(len(self.labels))]
            self.test_triplets = triplets
        

    def _get_clip_pair(self, idx):
        """ return (video, optflow, bbox, label) pair of an item """
        video, _, info, _ = self.video_clips.get_clip(idx)
        optflow, _, _, _ = self.optflow_clips.get_clip(idx)
        video = self._to_pil_image(video)
        optflow = self._to_pil_image(optflow)
        #label = self.class_to_idx[info['label']]
        label = info['label']
        bboxes = self._get_bounding_box(info)
        return (video, optflow, bboxes, label)
    
    
    def _make_item(self, idx):
        modes = ['anchor', 'pos', 'neg']
        
        if self._is_train:
            sample = {mode: {data: torch.tensor(0) 
                             for data in ['rgb', 'flow', 'bbox']} 
                             for mode in modes} # ['anchor', 'pos', 'neg']
            
            video, optflow, bboxes, label = self._get_clip_pair(idx)
            sample['anchor']['rgb'] = video
            sample['anchor']['flow'] = optflow
            sample['anchor']['bbox'] = bboxes
            sample['anchor']['label'] = label

            pos_idx = idx
            while pos_idx == idx:
                pos_idx = np.random.choice(self.label_to_indices[label])
                neg_label = np.random.choice(list(set(self.labels) - set([label])))
                neg_idx = np.random.choice(self.label_to_indices[neg_label])
                
                video, optflow, bboxes, label = self._get_clip_pair(pos_idx)
                sample['pos']['rgb'] = video
                sample['pos']['flow'] = optflow
                sample['pos']['bbox'] = bboxes
                sample['pos']['label'] = label
                video, optflow, bboxes, label = self._get_clip_pair(neg_idx)
                sample['neg']['rgb'] = video
                sample['neg']['flow'] = optflow
                sample['neg']['bbox'] = bboxes
                sample['neg']['label'] = label
        else:
            anchor_idx, pos_idx, neg_idx = self.test_triplets[idx]
            
            for i, idx_tmp in enumerate([anchor_idx, pos_idx, neg_idx]):
                video, optflow, bboxes, label = self._get_clip_pair(idx_tmp)
                sample[modes[i]]['rgb'] = video
                sample[modes[i]]['flow'] = optflow
                sample[modes[i]]['bbox'] = bboxes
                sample[modes[i]]['bbox'] = label
        return sample
    
    def __getitem__(self, idx):
        modes = ['anchor', 'pos', 'neg']
        sample = self._make_item(idx)
        
        for mode in modes:
            video = sample[mode]['rgb']
            optflow = sample[mode]['flow']
            rois = sample[mode]['bbox']
            label = sample[mode]['label']
            label = self.class_to_idx[label]
            
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

            if self.spatial_transform is not None:
                self.spatial_transform.randomize_parameters()
                video = [self.spatial_transform(img) for img in video]
                optflow = [self.spatial_transform(img) for img in optflow]

            
            video = torch.stack(video).transpose(0, 1) # TCHW-->CTHW
            optflow = torch.stack(optflow).transpose(0, 1) # TCHW-->CTHW
            optflow = optflow[:-1,:,:,:] # 3->2 channel
            label = torch.tensor(label)
            
            sample[mode]['rgb'] = video
            sample[mode]['flow'] = optflow
            sample[mode]['label'] = label
        
        return [(sample[mode]['rgb'], 
                 sample[mode]['flow'],
                 sample[mode]['label']) for mode in modes]
    
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
                num_frames[label] += len(v)
        return num_frames
    
    def _get_bounding_box(self, info, fixed_fov=True, upper_only=True, align=False, padding=False): #, label):
        
        person_id = info['target_id']
        bboxes = []
        for i in info['annotations']:
            bbox, _ = i[person_id]
            x1, y1, x2, y2 = bbox
            w = x2 - x1
            h = y2 - y1
            bboxes.append([x1, y1, w, h])

        if fixed_fov:
            bboxes = self.fix_field_of_view(bboxes)
        if upper_only:
            bboxes = [self.crop_upper_body_with_ratio(b) for b in bboxes]
        if align:
            bboxes = [self.align_boundingbox(b) for b in bboxes]
        if padding:
            bboxes = self._padding(bboxes)
        
        return bboxes
    
    def fix_field_of_view(self, bboxes):
        
        fov = [1280, 720, 0, 0]
        
        for bbox in bboxes:
            x1, y1, w, h = bbox
            x2 = x1 + w
            y2 = y1 + h
            
            if x1 < fov[0]:
                fov[0] = x1
            if y1 < fov[1]:
                fov[1] = y1
            if x2 > fov[2]:
                fov[2] = x2
            if y2 > fov[3]:
                fov[3] = y2
            
        x1, y1, x2, y2 = fov
        w = x2 - x1
        h = y2 - y1
        fov = [x1, y1, w, h] 
        return [fov for i in range(len(bboxes))]

    
    def crop_upper_body_with_ratio(self, bbox):
        x, y, w, h = bbox
        ratio = h/w
        if ratio > 1.5 :
            h /= (ratio-0.5)
        return [x, y, w, h]
    
    def align_boundingbox(self, bbox):
        x, y, w, h = bbox
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
#                 if flows is not None:
#                     flows = np.asarray(flows)
#                     write_video(v_output, flows, fps=15)
                    
    
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
        vname, person_ids, labels = self.samples[vidx]
        return (vidx, cidx)
    
    def _to_pil_image(self, video):
        video = [v.permute(2, 0, 1) for v in video] # for to_pil_image
        return [F.to_pil_image(img) for img in video]
        
    def __len__(self):
        return self.video_clips.num_clips()
    

    def _is_train(self):
        return True if os.path.basename(self.root) == 'train' else False
    
    
    
class BalancedBatchSampler(BatchSampler):
    """
    Taken from "https://github.com/adambielski/siamese-triplet/blob/master/datasets.py"
    BatchSampler - from a MNIST-like dataset, samples n_classes and within these classes samples n_samples.
    Returns batches of size n_classes * n_samples
    """

    def __init__(self, labels, n_classes, n_samples):
        self.labels = labels
        self.labels_set = self.labels
        self.label_to_indices = {label: np.where(np.array(self.labels) == label)[0]
                                             for label in set(self.labels)}
        for l in self.labels_set:
            np.random.shuffle(self.label_to_indices[l])
            
        self.used_label_indices_count = {label: 0 for label in self.labels_set}
        self.count = 0
        self.n_classes = n_classes
        self.n_samples = n_samples
        self.n_dataset = len(self.labels)
        self.batch_size = self.n_samples * self.n_classes

    def __iter__(self):
        self.count = 0
        while self.count + self.batch_size < self.n_dataset:
            classes = np.random.choice(self.labels_set, self.n_classes, replace=False)
            indices = []
            for class_ in classes:
                indices.extend(self.label_to_indices[class_][
                               self.used_label_indices_count[class_]:self.used_label_indices_count[
                                                                         class_] + self.n_samples])
                self.used_label_indices_count[class_] += self.n_samples
                if self.used_label_indices_count[class_] + self.n_samples > len(self.label_to_indices[class_]):
                    np.random.shuffle(self.label_to_indices[class_])
                    self.used_label_indices_count[class_] = 0
            yield indices
            self.count += self.n_classes * self.n_samples

    def __len__(self):
        return self.n_dataset // self.batch_size