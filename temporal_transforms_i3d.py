import random
import math
import logging 

class LoopPadding(object):
    
    def __init__(self, size):
        self.size = size

    def __call__(self, frame_indices):
        out = frame_indices
        out = self.__sizecheck__(out)
        out = self.loop(out)
        return out
    
    def loop(self, out):
        for index in out:
            if len(out) >= 16:
                break
            out.append(index)
        return out
    
    def __sizecheck__(self, out):
        if len(out) >= self.size:
            transforms = TemporalRandomChoice([
                TemporalBeginCrop(self.size),
                TemporalRandomCrop(self.size),
                TemporalCenterCrop(self.size)])
            out = transforms(out)
        return out


class MirrorPadding(LoopPadding):
    
    def __init__(self, size):
        super(MirrorPadding, self).__init__(size)
        self.size = size

    def __call__(self, frame_indices):
        out = frame_indices[::-1]
        out = self.__sizecheck__(out)
        out = self.loop(out)
        return out
    
    
class MirrorLoopPadding(LoopPadding):
    
    def __init__(self, size):
        super(MirrorLoopPadding, self).__init__(size)
        self.size = size

    def __call__(self, frame_indices):
        #print("mirror loop padding...")
        out = frame_indices
        out = self.__sizecheck__(out)
        
        for i in range(100):
            out += self.__getmirror__(frame_indices, i)
            
            if len(out) >= self.size:
                out = out[:self.size]
                break
        return out
    
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

    def __call__(self, frame_indices):
        #print("begin cropping ...")
        out = frame_indices[:self.size]
        out = self.__sizecheck__(out)
        return out
    
    def __sizecheck__(self, out):
        if len(out) < self.size:
            transforms = TemporalRandomChoice([
                LoopPadding(self.size),
                MirrorPadding(self.size),
                MirrorLoopPadding(self.size)])
            out = transforms(out)
        return out


class TemporalCenterCrop(object):
    """Temporally crop the given frame indices at a center.
    If the number of frames is less than the size,
    loop the indices as many times as necessary to satisfy the size.
    Args:
        size (int): Desired output size of the crop.
    """

    def __init__(self, size):
        self.size = size

    def __call__(self, frame_indices):
        """
        Args:
            frame_indices (list): frame indices to be cropped.
        Returns:
            list: Cropped frame indices.
        """
        
        center_index = len(frame_indices) // 2
        begin_index = max(0, center_index - (self.size // 2))
        end_index = min(begin_index + self.size, len(frame_indices))

        out = frame_indices[begin_index:end_index]
        out = self.__sizecheck__(out)
        
        return out
    
    def __sizecheck__(self, out):
        if len(out) < self.size:
            transforms = TemporalRandomChoice([
                LoopPadding(self.size),
                MirrorPadding(self.size),
                MirrorLoopPadding(self.size)])
            out = transforms(out)
        return out
    

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

    def __call__(self, frame_indices):
        """
        Args:
            frame_indices (list): frame indices to be cropped.
        Returns:
            list: Cropped frame indices.
        """

        #print("random cropping ...")
        rand_end = max(0, len(frame_indices) - self.size - 1)
        begin_index = random.randint(0, rand_end)
        end_index = min(begin_index + self.size, len(frame_indices))

        out = frame_indices[begin_index:end_index]
        out = self.__sizecheck__(out)
        
        return out
    
    def __sizecheck__(self, out):
        if len(out) < self.size:
            transforms = TemporalRandomChoice([
                LoopPadding(self.size),
                MirrorPadding(self.size),
                MirrorLoopPadding(self.size)])
            out = transforms(out)
        return out

    
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

    def __call__(self, frame_indices):
        if self.p < random.random():
            return frame_indices
        
        for t in self.transforms:
            frame_indices = t(frame_indices)
        return frame_indice

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
    def __call__(self, frame_indices):
        t = random.choice(self.transforms)
        logging.info(str(t))
        return t(frame_indices)
    
    
    
