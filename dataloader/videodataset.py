import torch
from .video_utils import VideoClips
from torchvision.datasets.utils import list_dir
from torchvision.datasets.folder import has_file_allowed_extension
from torchvision.datasets.vision import VisionDataset
from torchvision.transforms import functional as F

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


class VideoDataset(VisionDataset):

    def __init__(self, root, frames_per_clip, 
                 step_between_clips=1, 
                 frame_rate=None,
                 spatial_transform=None,
                 temporal_transform=None):
        super(VideoDataset, self).__init__(root)
        extensions = ('',)
        classes = list(sorted(list_dir(root)))
        class_to_idx = {classes[i]: i for i in range(len(classes))}
        self.samples = make_dataset(self.root, class_to_idx, extensions, is_valid_file=None)
        self.classes = classes
        video_list = [x[0] for x in self.samples]
        video_list = [x for x in video_list if 'reverse' not in x] ## TODO
        self.video_clips = VideoClips(video_list, frames_per_clip, step_between_clips, frame_rate)
        print('Number of {} video clips: {:d}'.format(root, self.video_clips.num_clips()))
        #self.transform = transform
        self.spatial_transform = spatial_transform
        self.temporal_transform = temporal_transform
    
    
    def __getitem__(self, idx):
        """
            video (Tensor[T, H, W, C]): the `T` video frames
            label (int): class of the video clip
        """
        video, _, _, video_idx = self.video_clips.get_clip(idx)
        video = self._to_pil_image(video)
        label = self.samples[video_idx][1]
        
        if self.temporal_transform is not None:
            video = [self.temporal_transform(img) for img in video]
        
        if self.spatial_transform is not None:
            self.spatial_transform.randomize_parameters()
            video = [self.spatial_transform(img) for img in video]
            
        video = torch.stack(video).transpose(0, 1) # TCHW-->CTHW
        label = torch.tensor(label).unsqueeze(0) # () -> (1,)
        return video, label
    
    def _to_pil_image(self, video):
        video = [v.permute(2, 0, 1) for v in video] # for to_pil_image
        return [F.to_pil_image(img) for img in video]
    
    def __len__(self):
        return self.video_clips.num_clips()
    
    def _get_clip_loc(self, idx):
        vidx, cidx = self.video_clips.get_clip_location(idx)
        vname, label = self.samples[vidx]
        return (vidx, cidx)
