import torch
from .video_utils import VideoClips
from torchvision.datasets.utils import list_dir
from torchvision.datasets.folder import has_file_allowed_extension
from torchvision.datasets.vision import VisionDataset
from torchvision.transforms import functional as F


def make_dataset(dir, class_to_idx, extensions=None, is_valid_file=None):
    """
    Arguments:
        dir: phase (train, val, test)
        target: class name
        root: video clip name (ex. (vid)_(vdate)_frames000000)
    """
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


class VideoDataset(VisionDataset):
    
    def __init__(self, root, frames_per_clip, 
                 step_between_clips=1, 
                 frame_rate=None,
                 downsample_size=None,
                 spatial_transform=None,
                 temporal_transform=None,
                 target_transform=None):
        
        super(VideoDataset, self).__init__(root)
        extensions = ('',)
        classes = list(sorted(list_dir(root)))
        class_to_idx = {classes[i]: i for i in range(len(classes))}
        
        self.samples = make_dataset(self.root, class_to_idx, extensions, is_valid_file=None)
        self.classes = classes
        
        self.frames_per_clip = frames_per_clip
        self.steps = self._init_steps(step_between_clips)
        self.video_list = [x[0] for x in self.samples]
        
        self.video_clips = VideoClips(self.video_list, 
                                      frames_per_clip, 
                                      step_between_clips, 
                                      frame_rate,
                                      downsample_size=downsample_size)
        
        #self.transform = transform
        self.spatial_transform = spatial_transform
        self.temporal_transform = temporal_transform
        
        print('Number of {} video clips: {:d} ({:d} images)'.format(
            root, self.video_clips.num_clips(), self.video_clips.num_total_frames()))
        print("Number of clips per class: ", self._num_clips_per_class())
        print("Number of frames per class: ", self._num_frames_per_class())
    
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
            vidx, _ = self.video_clips.get_clip_location(i)
            label = self.samples[vidx][1]
            num_clips[classes[label]] += 1
        return num_clips
    
    def _num_frames_per_class(self):
        classes = self.classes
        class_to_idx = {classes[i]: i for i in range(len(classes))}
        num_frames = {cls:0 for cls in classes}
        
        for path in self.video_clips.video_paths:
            length = self.video_clips.num_image_frames(path)
            _, _, label, _ = self.video_clips._split_path(path)
            label = class_to_idx[label]
            num_frames[classes[label]] += length
        return num_frames
    
    def _make_item(self, idx):
        video, _, info, video_idx = self.video_clips.get_clip(idx)
        video = self._to_pil_image(video)
        label = self.samples[video_idx][1]
        return (video, label)
    
    def __getitem__(self, idx):
        """
            video (Tensor[T, H, W, C]): the `T` video frames
            label (int): class of the video clip
        """
#         print("videodataset, __getitem__", 
#               [data[0].size() if data is not None else None for data in self.video_clips.shared_data])
        video, label = self._make_item(idx)
        if self.temporal_transform is not None:
            self.temporal_transform.randomize_parameters()
            video = self.temporal_transform(video)
            
        if self.spatial_transform is not None:
            self.spatial_transform.randomize_parameters()
            video = [self.spatial_transform(img) for img in video]
        
        if self.target_transform is not None:
            label = self.target_transform(label)
    
        video = torch.stack(video).transpose(0, 1) # TCHW-->CTHW
        #TODO:
        #label = _binary(label)
        label = torch.tensor(label)
        
        if len(self.classes) < 2:
            label = label.unsqueeze(0) # () -> (1,) for binary classification
        
        return video, label
    
    def _get_clip_loc(self, idx):
        vidx, cidx = self.video_clips.get_clip_location(idx)
        vname, label = self.samples[vidx]
        return (vidx, cidx)
    
    def _to_pil_image(self, video):
        video = [v.permute(2, 0, 1) for v in video] # for to_pil_image
        return [F.to_pil_image(img) for img in video]
        
    def __len__(self):
        return self.video_clips.num_clips()
    