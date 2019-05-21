import random
import math
import numbers
import collections
import numpy as np
import torch

from spatial_transforms import *

class TemporalCompose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, flow):
        for t in self.transforms:
            img, flow = t(img, flow)
        return img, flow

    def randomize_parameters(self):
        for t in self.transforms:
            t.randomize_parameters()
            
class TemporalColorJitter(ColorJitter):
    def __init__(self, brightness=0, contrast=0, saturation=0, hue=0):
        super(TemporalColorJitter, self).__init__(brightness, contrast, saturation, hue)
        self.brightness = self._check_input(brightness, 'brightness')
        self.contrast = self._check_input(contrast, 'contrast')
        self.saturation = self._check_input(saturation, 'saturation')
        self.hue = self._check_input(hue, 'hue', center=0, bound=(-0.5, 0.5),
                                     clip_first_on_zero=False)
        
    def __call__(self, imgs, flows):
        transform = self.get_params(self.brightness, self.contrast,
                                    self.saturation, self.hue)
        out_imgs=[]
        out_flows=[]
        for img, flow in zip(imgs, flows):
            out_imgs.append(transform(img))
            out_flows.append(transform(flow))
        return out_imgs, out_flows
    
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
        
    def __call__(self, imgs, flows):
        angle = self.get_params(self.degrees)
        out_imgs=[]
        out_flows=[]
        for img, flow in zip(imgs, flows):
            out_imgs.append(img.rotate(angle, self.resample, self.expand, self.center))
            out_flows.append(flow.rotate(angle, self.resample, self.expand, self.center)) 
        return out_imgs, out_flows
