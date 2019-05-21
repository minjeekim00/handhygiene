import random
import math
import numbers
import collections
import numpy as np
import torch

from PIL import Image, ImageOps
try:
    import accimage
except ImportError:
    accimage = None

from spatial_transforms import *
    
class TemporalRandomRotation(RandomRotation):
    
    def __init__(self, degrees, resample=False, expand=False, center=None):
        super(TemporalRandomRotation, self).__init__(degrees, resample, expand, center)
        if isinstance(degrees, numbers.Number):
            if degrees < 0:
                raise ValueError("If degrees is a single number, it must be positive.")
            self.degrees = (-degrees, degrees)
        else:
            if len(degrees) != 2:
                raise ValueError("If degrees is a sequence, it must be of len 2.")
            self.degrees = degrees
        
    def __call__(self, imgs):
        angle = self.get_params(self.degrees)
        out_imgs=[]
        for img in imgs:
            out_imgs.append(img.rotate(angle, self.resample, self.expand, self.center))     
        return out_imgs
            
    def randomize_parameters(self):
        pass
