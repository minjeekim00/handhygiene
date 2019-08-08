import os
import re
import gc
import torch
import numpy as np
from glob import glob
from PIL import Image
    
from torchvision.datasets.folder import has_file_allowed_extension
from torchvision.datasets.folder import is_image_file
from torchvision.datasets.folder import IMG_EXTENSIONS


def get_frames(dirname):
    return sorted([os.path.join(dirname, file) 
                   for file in os.listdir(dirname) 
                   if is_image_file(file)])

def read_video(dirname, start_pts=0, end_pts=None):
    
    frames = get_frames(dirname)
    video = []
    for i, frame in enumerate(frames):
        with open(frame, 'rb') as f:
            img = Image.open(f)
            img = img.convert('RGB')
            img = np.asarray(img)
            video.append(img)
            
    if end_pts is None:
        end_pts = len(video)

    if end_pts < start_pts:
        raise ValueError("end_pts should be larger than start_pts, got "
                         "start_pts={} and end_pts={}".format(start_pts, end_pts))
        
    video = np.asarray(video)
    video = torch.tensor(video)
    return read_video_as_clip(video, start_pts, end_pts)

def read_video_as_clip(video, start_pts, end_pts):
    video = video[start_pts:end_pts+1]
    audio = torch.tensor([]) #tmp
    info = {'video_fps': 15.0}
    return (video, audio, info)


def read_video_timestamps(dirname):
    """ tmp function """
    frames = get_frames(dirname)
    return (list(range(len(frames)*1)), 15.0)
    
    
def write_video(filename, video_array, fps, video_codec='libx264', options=None):
    """
    Writes a 4d tensor in [T, H, W, C] format in a video file
    Parameters
    ----------
    filename : str
        path where the video will be saved
    video_array : Tensor[T, H, W, C]
        tensor containing the individual frames, as a uint8 tensor in [T, H, W, C] format
    fps : Number
        frames per second
    """
    
    try:
        import av
        av.logging.set_level(av.logging.ERROR)
    except ImportError:
        av = None


    def _check_av_available():
        if av is None:
            raise ImportError("""\
    PyAV is not installed, and is necessary for the video operations in torchvision.
    See https://github.com/mikeboers/PyAV#installation for instructions on how to
    install PyAV on your system.
    """)
        
    _check_av_available()
    video_array = torch.as_tensor(video_array, dtype=torch.uint8).numpy()

    container = av.open(filename, mode='w')

    stream = container.add_stream(video_codec, rate=fps)
    stream.width = video_array.shape[2]
    stream.height = video_array.shape[1]
    stream.pix_fmt = 'yuv420p' if video_codec != 'libx264rgb' else 'rgb24'
    stream.options = options or {}

    for img in video_array:
        frame = av.VideoFrame.from_ndarray(img, format='rgb24')
        frame.pict_type = 'NONE'
        for packet in stream.encode(frame):
            container.mux(packet)

    # Flush stream
    for packet in stream.encode():
        container.mux(packet)

    # Close the file
    container.close()
    
