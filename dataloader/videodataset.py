import torch
from torchvision.datasets.video_utils import VideoClips
from torchvision.datasets.utils import list_dir
from torchvision.datasets.folder import make_dataset
from torchvision.datasets.vision import VisionDataset

class VideoDataset(VisionDataset):

    def __init__(self, root, frames_per_clip, step_between_clips=1, 
                 spatial_transform=None,
                 temporal_transform=None):
        super(HandHygiene, self).__init__(root)
        extensions = ('mp4',)

        classes = list(sorted(list_dir(root)))
        class_to_idx = {classes[i]: i for i in range(len(classes))}
        self.samples = make_dataset(self.root, class_to_idx, extensions, is_valid_file=None)
        self.classes = classes
        video_list = [x[0] for x in self.samples]
        self.video_clips = VideoClips(video_list, frames_per_clip, step_between_clips)
        #self.transform = transform
        self.spatial_transform = spatial_transform
        self.temporal_transform = temporal_transform
        

    def __len__(self):
        return self.video_clips.num_clips()

    def __getitem__(self, idx):
        """
            video (Tensor[T, H, W, C]): the `T` video frames
            label (int): class of the video clip
        """
        video, _, _, video_idx = self.video_clips.get_clip(idx)
        label = self.samples[video_idx][1]

        video = [v.permute(2, 0, 1) for v in video] # for to_pil_image
        if self.spatial_transform is not None:
            self.spatial_transform.randomize_parameters()
            video = [self.spatial_transform(img) for img in video]

        video = torch.stack(video).transpose(0, 1) # TCHW-->CTHW
        label = torch.tensor(label).unsqueeze(0) # () -> (1,)
        return video, label