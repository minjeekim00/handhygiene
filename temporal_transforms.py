import random
import math
import logging 
from datetime import datetime

class LoopPadding(object):
    
    def __init__(self, size):
        self.size = size

    def __call__(self, video):
        #video = self.__sizecheck__(video)
        video = self.loop(video)
        return video
    
    def loop(self, video):
        for index in video:
            if len(video) >= self.size:
                break
            video.append(index)
        return video
    
    def __sizecheck__(self, video):
        if len(video) >= self.size:
            transforms = TemporalRandomChoice([
                TemporalBeginCrop(self.size),
                TemporalRandomCrop(self.size),
                TemporalCenterCrop(self.size)])
            video = transforms(video)
        return video
    
    def randomize_parameters(self):
        random.seed(datetime.now())

class MirrorPadding(LoopPadding):
    
    def __init__(self, size):
        super(MirrorPadding, self).__init__(size)
        self.size = size

    def __call__(self, video):
        video = video[::-1]
        video = self.loop(video)
        return video
    
    
class MirrorLoopPadding(LoopPadding):
    
    def __init__(self, size):
        super(MirrorLoopPadding, self).__init__(size)
        self.size = size

    def __call__(self, video):
        for i in range(100):
            video += self.__getmirror__(video, i)
            
            if len(video) >= self.size:
                video = video[:self.size]
                break
        return video
    
    def __getmirror__(self, li, i):
        return list(reversed(li))[1:] if i == 0 or i/2 == 1 else li
    
    
        
    

class TemporalBeginCrop(object):
    """Temporally crop the given frame indices at a beginning.
    If the number of frames is less than the size,
    loop the indices as many times as necessary to satisfy the size.
    Args:
        size (int): Desired output size of the crop.
    """

    def __init__(self, size):
        self.size = size

    def __call__(self, video):
        return video[:self.size]
    
    

class TemporalCenterCrop(object):
    """Temporally crop the given frame indices at a center.
    If the number of frames is less than the size,
    loop the indices as many times as necessary to satisfy the size.
    Args:
        size (int): Desired output size of the crop.
    """

    def __init__(self, size):
        self.size = size

    def __call__(self, video):
        """
        Args:
            video (list): PIL Images (frames) to be cropped.
        Returns:
            list: Cropped frame indices.
        """
        
        center_index = len(video) // 2
        begin_index = max(0, center_index - (self.size // 2))
        end_index = min(begin_index + self.size, len(video))

        video = video[int(begin_index):int(end_index)]
        return video
    

class TemporalRandomCrop(object):
    """
    Temporally crop the given frame indices at a random location.
    If the number of frames is less than the size,
    loop the indices as many times as necessary to satisfy the size.
    Args:
        size (int): Desired output size of the crop.
    """

    def __init__(self, size):
        self.size = size

    def __call__(self, video):
        """
        Args:
            video (list): frames to be cropped.
        Returns:
            list: Cropped frame indices.
        """

        #print("random cropping ...")
        rand_end = max(0, len(video) - self.size - 1)
        begin_index = random.randint(0, rand_end)
        end_index = min(begin_index + self.size, len(video))

        video = video[int(begin_index):int(end_index)]
        return video
    
    def randomize_parameters(self):
        random.seed(datetime.now())
    
    
class RandomTransforms(object):
    """Base class for a list of transformations with randomness
    Args:
        transforms (list or tuple): list of transformations
    """

    def __init__(self, transforms):
        assert isinstance(transforms, (list, tuple))
        self.transforms = transforms

    def __call__(self, *args, **kwargs):
        raise NotImplementedError()

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        for t in self.transforms:
            format_string += '\n'
            format_string += '    {0}'.format(t)
        format_string += '\n)'
        return format_string
    
    
class TemporalRandomApply(RandomTransforms):
    """Apply randomly a list of transformations with a given probability
    Args:
        transforms (list or tuple): list of transformations
        p (float): probability
    """

    def __init__(self, transforms, p=0.5):
        super(TemporalRandomApply, self).__init__(transforms)
        self.p = p

    def __call__(self, video):
        if self.p < random.random():
            return video
        
        for t in self.transforms:
            video = t(video)
        return video

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        format_string += '\n    p={}'.format(self.p)
        for t in self.transforms:
            format_string += '\n'
            format_string += '    {0}'.format(t)
        format_string += '\n)'
        return format_string
    
    
class TemporalRandomChoice(RandomTransforms):
    """Apply single transformation randomly picked from a list
    """
    def __call__(self, video):
        t = random.choice(self.transforms)
        #logging.info(str(t))
        return t(video)
    
    def randomize_parameters(self):
        random.seed(datetime.now())