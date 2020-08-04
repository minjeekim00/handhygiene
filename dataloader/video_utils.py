## https://github.com/pytorch/vision/blob/master/torchvision/datasets/video_utils.py

import os
import bisect
import math
import torch
from dataloader.io.video import read_video_timestamps
from dataloader.io.video import read_video_as_clip
from dataloader.io.video import read_video
from dataloader.io.video import get_frames
from dataloader.io.video import _get_bbox_info
#from torchvision.datasets.utils import tqdm
from tqdm import tqdm


def unfold(tensor, size, step, dilation=1):
    assert tensor.dim() == 1
    o_stride = tensor.stride(0)
    numel = tensor.numel()
    new_stride = (step * o_stride, dilation * o_stride)
    new_size = ((numel - (dilation * (size - 1) + 1)) // step + 1, size)
    if new_size[0] < 1:
        #new_size = (0, size)
        #not to exclude video shorther than size
        return tensor.unsqueeze(0)
    return torch.as_strided(tensor, new_size, new_stride)


class VideoClips(object):
    def __init__(self, video_paths, clip_length_in_frames=16, frames_between_clips=1,
                 frame_rate=None, _precomputed_metadata=None, with_detection=False,
                  downsample_size=None, annotation=None, target_classes=None):
        self.video_paths = video_paths
        self.annotation = annotation
        self.target_classes = target_classes
        if _precomputed_metadata is None:
            self._compute_frame_pts()
        else:
            self._init_from_metadata(_precomputed_metadata)
        self.compute_clips(clip_length_in_frames, frames_between_clips, frame_rate, self.target_classes)
        
        self.downsample_size = downsample_size

        self.shared_manager = None
        self.shared_data = [None for video_path in self.video_paths]
        self.with_detection = with_detection

    def _compute_frame_pts(self):
        self.video_pts = []
        self.video_fps = []
        class DS(object):
            def __init__(self, x):
                self.x = x

            def __len__(self):
                return len(self.x)

            def __getitem__(self, idx):
                return read_video_timestamps(self.x[idx])

        import torch.utils.data
        dl = torch.utils.data.DataLoader(
            DS(self.video_paths),
            batch_size=16,
            num_workers=torch.get_num_threads(),
            collate_fn=lambda x: x)

        with tqdm(total=len(dl)) as pbar:
            for batch in dl:
                pbar.update(1)
                clips, fps = list(zip(*batch))
                clips = [torch.as_tensor(c) for c in clips]
                self.video_pts.extend(clips)
                self.video_fps.extend(fps)

    def _init_from_metadata(self, metadata):
        assert len(self.video_paths) == len(metadata["video_pts"])
        assert len(self.video_paths) == len(metadata["video_fps"])
        self.video_pts = metadata["video_pts"]
        self.video_fps = metadata["video_fps"]

    def subset(self, indices):
        video_paths = [self.video_paths[i] for i in indices]
        video_pts = [self.video_pts[i] for i in indices]
        video_fps = [self.video_fps[i] for i in indices]
        metadata = {
            "video_pts": video_pts,
            "video_fps": video_fps
        }
        return type(self)(video_paths, self.num_frames, self.steps, self.frame_rate,
                          _precomputed_metadata=metadata)

    @staticmethod
    def compute_clips_for_video(video_pts, num_frames, step, fps, frame_rate):
        if fps is None:
            # if for some reason the video doesn't have fps (because doesn't have a video stream)
            # set the fps to 1. The value doesn't matter, because video_pts is empty anyway
            fps = 1
        if frame_rate is None:
            frame_rate = fps
        total_frames = len(video_pts) * (float(frame_rate) / fps)
        idxs = VideoClips._resample_video_idx(int(math.floor(total_frames)), fps, frame_rate)
        video_pts = video_pts[idxs]
        clips = unfold(video_pts, num_frames, step)
        if isinstance(idxs, slice):
            idxs = [idxs] * len(clips)
        else:
            idxs = unfold(idxs, num_frames, step)
        return clips, idxs

    def compute_clips(self, num_frames, steps, frame_rate=None, target_classes=[]):
        """
        Compute all consecutive sequences of clips from video_pts.
        Always returns clips of size `num_frames`, meaning that the
        last few frames in a video can potentially be dropped.
        Arguments:
            num_frames (int): number of frames for the clip
            step (int): distance between two clips
            dilation (int): distance between two consecutive frames
                in a clip
        """
        self.num_frames = num_frames
        self.steps = steps
        self.frame_rate = frame_rate
        self.clips = []
        self.resampling_idxs = []
        self.info = []
        for vidx, (video_pts, fps) in enumerate(zip(self.video_pts, self.video_fps)):
            
            # customize step per class
            if True: #self.with_detection:
                # TODO
                frame = get_frames(self.video_paths[vidx])[0]
                image_path = os.path.splitext(os.path.basename(frame))[0]
                bbox = _get_bbox_info(image_path, self.annotation, target_classes)
                
                for i, (k, v )in enumerate(bbox.items()):
                    label = v[1]
                    step = steps[label]
                    clips, idxs = self.compute_clips_for_video(video_pts, num_frames, step, fps, frame_rate)
                    
                    self.clips.append(clips)
                    self.resampling_idxs.append(idxs)
                    
                    for cidx in range(len(clips)):
                        #TODO:
                        self.info.append([vidx, i, k, cidx, label])
            else:
                # get clip info
                _, _, label, name = self._split_path(self.video_paths[vidx])
                step = steps[label]
                clips, idxs = self.compute_clips_for_video(video_pts, num_frames, step, fps, frame_rate)
                self.clips.append(clips)
                self.resampling_idxs.append(idxs)
            
        clip_lengths = torch.as_tensor([len(v) for v in self.clips])
        self.cumulative_sizes = clip_lengths.cumsum(0).tolist()
        
    def _split_path(self, path):
        if 'flow' in path: # for optical flow dir
            path = os.path.split(path)[0]
        
        root, name = os.path.split(path)
        root, label = os.path.split(root)
        root, phase = os.path.split(root)
        return root, phase, label, name
#         return root, phase, name
        
    def __len__(self):
        return self.num_clips()

    def num_videos(self):
        return len(self.video_paths)

    def num_clips(self):
        """
        Number of subclips that are available in the video list.
        """
        return self.cumulative_sizes[-1]
    
    def num_total_frames(self):
        return sum([len(get_frames(path)) for path in self.video_paths])
    
    def num_image_frames(self, path):
        return len(get_frames(path))

    def get_clip_location(self, idx):
        """
        Converts a flattened representation of the indices into a video_idx, clip_idx
        representation.
        """
        '''
        video_idx = bisect.bisect_right(self.cumulative_sizes, idx)
        if video_idx == 0:
            clip_idx = idx
        else:
            clip_idx = idx - self.cumulative_sizes[video_idx - 1]
        return video_idx, clip_idx  
        '''
        return self.info[idx], bisect.bisect_right(self.cumulative_sizes, idx)
                                            
    def get_video_idx(self, idx):
        return bisect.bisect_right(self.cumulative_sizes, idx)
    
    @staticmethod
    def _resample_video_idx(num_frames, original_fps, new_fps):
        step = float(original_fps) / new_fps
        if step.is_integer():
            # optimization: if step is integer, don't need to perform
            # advanced indexing
            step = int(step)
            return slice(None, None, step)
        idxs = torch.arange(num_frames, dtype=torch.float32) * step
        idxs = idxs.floor().to(torch.int64)
        return idxs

    def cache_video(self, video_idx, video):
        """ cache video """
        if 'flow' in self.video_paths[video_idx]:
            print("[Flow] Filling cache for index {}".format(video_idx))
        else:
            print("[RGB] Filling cache for index {}".format(video_idx))
        self.shared_data[video_idx] = video

    def get_video(self, video_idx):
        """ get video from shared data """
        video_path = self.video_paths[video_idx]
        
        if self.shared_data[video_idx] is None:
            video = read_video(video_path, 0, None, self.with_detection, self.downsample_size, self.annotation)
            self.cache_video(video_idx, video)
                
        return self.shared_data[video_idx]
    
    def get_clip(self, idx):
        if idx >= self.num_clips():
            raise IndexError("Index {} out of range "
                             "({} number of clips)".format(idx, self.num_clips()))
        #video_idx, clip_idx = self.get_clip_location(idx)
        (video_idx, idx, person_idx, clip_idx, label), video_c = self.get_clip_location(idx)
        video_path = self.video_paths[video_idx]
        
        #print("video_idx {}, idx {}, video_c {}, clip_idx {}".format(video_idx, idx, video_c, clip_idx))
        clip_pts = self.clips[video_c][clip_idx]
        start_pts = clip_pts[0].item()
        end_pts = clip_pts[-1].item()
        
        video_raw = self.get_video(video_idx)
        video, audio, info = read_video_as_clip(video_raw, start_pts, end_pts, self.with_detection, target=person_idx)
        
        if self.frame_rate is not None:
            #TODO:
            resampling_idx = self.resampling_idxs[video_idx][clip_idx]
            if isinstance(resampling_idx, torch.Tensor):
                resampling_idx = resampling_idx - resampling_idx[0]
            video = video[resampling_idx]
            info["video_fps"] = self.frame_rate
        #if len(video) < self.num_frames:
        #    print("{} x {}".format(video.shape, self.num_frames))
        info['target_id'] = person_idx
        info['label']= label
        return video, audio, info, video_idx

#     def cache_video_all(self, manager):
#         self.shared_data = manager.list([self.get_video(video_idx) for video_idx 
#                             in tqdm(range(len(self.video_paths)))])
    
    def cache_video_all(self):
        self.shared_data = [self.get_video(video_idx) for video_idx 
                            in tqdm(range(len(self.video_paths)))]
    
    def set_shared_manager(self, manager):
        self.shared_manager = manager
        print("setting manager....")
        self.shared_data = self.shared_manager.list([None for video_path in self.video_paths])
        print("mp manager initialized....")