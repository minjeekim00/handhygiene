import random
import math
from train import logger

class LoopPadding(object):
    
    def __init__(self, size):
        self.size = size

    def __call__(self, frame_indices, coords):
        out = frame_indices
        out_crd = {'torso': coords['torso'],
                   'people': coords['people']}
        
        out, out_crd = self.__sizecheck__(out, out_crd)
        
        for i, index in enumerate(out):
            if len(out) >= self.size:
                break
            out.append(index)
            #out_crd['torso'].append(coords['torso'][i])
            out_crd['people'].append(coords['people'][i])
        return (out, out_crd)
                
    def __sizecheck__(self, out, out_crd):
        if len(out) >= self.size:
            print("size check: {}/{}".format(len(out), self.size))
            transforms = TemporalRandomChoice([
                TemporalBeginCrop(self.size),
                TemporalRandomCrop(self.size),
                TemporalCenterCrop(self.size)])
            out, out_crd = transforms(out, out_crd)
        return out, out_crd


class MirrorPadding(LoopPadding):
    
    def __init__(self, size):
        super(MirrorPadding, self).__init__(size)
        self.size = size

    def __call__(self, frame_indices, coords):
        
        ## for logging
        print("MirrorPadding")
        logger.write("MirrorPadding")
        
        out = frame_indices[::-1]
        out_crd = {'torso': coords['torso'][::-1],
                   'people': coords['people'][::-1]}
        
        out, out_crd = self.__sizecheck__(out, out_crd)
        
        for i, index in enumerate(out):
            if len(out) >= self.size:
                break
            out.append(index)
            #out_crd['torso'].append(coords['torso'][i])
            out_crd['people'].append(coords['people'][i])
        return (out, out_crd)
    
    
class MirrorLoopPadding(LoopPadding):
    
    def __init__(self, size):
        super(MirrorLoopPadding, self).__init__(size)
        self.size = size

    def __call__(self, frame_indices, coords):
        #print("mirror loop padding...")
        out = frame_indices
        out_crd = {'torso': coords['torso'],
                   'people': coords['people']}
        
        out, out_crd = self.__sizecheck__(out, out_crd)
        
        for i in range(100):
            out += self.__getmirror__(frame_indices, i)
            #out_crd['torso'] += self.__getmirror__(coords['torso'], i)
            out_crd['people'] += self.__getmirror__(coords['people'], i)
            
            if len(out) >= self.size:
                out = out[:self.size]
                #out_crd['torso'] = out_crd['torso'][:self.size]
                out_crd['people'] = out_crd['people'][:self.size]
                break
            
        return (out, out_crd)
    
    
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

    def __call__(self, frame_indices, coords):
        
        ## for logging
        logger.write("TemporalBeginCrop")
        
        #print("begin cropping ...")
        out = frame_indices[:self.size]
        out_crd = {'torso': coords['torso'],
                   'people': coords['people']}
        
        #out_crd['torso'] = out_crd['torso'][:self.size]
        out_crd['people'] = out_crd['people'][:self.size]
        out, out_crd = self.__sizecheck__(out, out_crd)
        
        return (out, out_crd)
    
    def __sizecheck__(self, out, out_crd):
        if len(out) < self.size:
            print("size check: {}/{}".format(len(out), self.size))
            transforms = TemporalRandomChoice([
                LoopPadding(self.size),
                MirrorPadding(self.size),
                MirrorLoopPadding(self.size)])
            out, out_crd = transforms(out, out_crd)
        return out, out_crd


class TemporalCenterCrop(object):
    """Temporally crop the given frame indices at a center.
    If the number of frames is less than the size,
    loop the indices as many times as necessary to satisfy the size.
    Args:
        size (int): Desired output size of the crop.
    """

    def __init__(self, size):
        self.size = size

    def __call__(self, frame_indices, coords):
        """
        Args:
            frame_indices (list): frame indices to be cropped.
        Returns:
            list: Cropped frame indices.
        """
        
        #print("center cropping ...")
        ## for logging
        logger.write("TemporalCenterCrop")
        
        center_index = len(frame_indices) // 2
        begin_index = max(0, center_index - (self.size // 2))
        end_index = min(begin_index + self.size, len(frame_indices))

        out = frame_indices[begin_index:end_index]
        out_crd = {'torso': coords['torso'],
                   'people': coords['people']}
        #out_crd['torso'] = out_crd['torso'][begin_index:end_index]
        out_crd['people'] = out_crd['people'][begin_index:end_index]
        out, out_crd = self.__sizecheck__(out, out_crd)
        
        return (out, out_crd)
    
    def __sizecheck__(self, out, out_crd):
        if len(out) < self.size:
            print("size check: {}/{}".format(len(out), self.size))
            transforms = TemporalRandomChoice([
                LoopPadding(self.size),
                MirrorPadding(self.size),
                MirrorLoopPadding(self.size)])
            out, out_crd = transforms(out, out_crd)
        return out, out_crd
    

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

    def __call__(self, frame_indices, coords):
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
        out_crd = {'torso': coords['torso'],
                   'people': coords['people']}
        
        #out_crd['torso'] = out_crd['torso'][begin_index:end_index]
        out_crd['people'] = out_crd['people'][begin_index:end_index]
        out, out_crd = self.__sizecheck__(out, out_crd)
        
        return (out, out_crd)
    
    def __sizecheck__(self, out, out_crd):
        if len(out) < self.size:
            transforms = TemporalRandomChoice([
                LoopPadding(self.size),
                MirrorPadding(self.size),
                MirrorLoopPadding(self.size)])
            out, out_crd = transforms(out, out_crd)
        return out, out_crd

    
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

    def __call__(self, frame_indices, coords):
        if self.p < random.random():
            return frame_indices, coords
        
        for t in self.transforms:
            frame_indices, coords = t(frame_indices, coords)
        return frame_indices, coords

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
    def __call__(self, frame_indices, coords):
        t = random.choice(self.transforms)
        return t(frame_indices, coords)
    
    
    
