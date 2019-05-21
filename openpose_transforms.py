import json
import os
import random
import math
import numbers
import collections
import numpy as np
import pandas as pd
from glob import glob
from tqdm import tqdm
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import torch
from PIL import Image, ImageOps
try:
    import accimage
except ImportError:
    accimage = None
    


class CropTorso(object):
    
    def __init__(self, size, interpolation=Image.BILINEAR):
        assert isinstance(size,
                          int) or (isinstance(size, collections.Iterable) and
                                   len(size) == 2)
        self.size = size
        self.interpolation = interpolation

    def __call__(self, img, flow, coords, index):
        """
        Args:
            img (PIL.Image): Image to be scaled.
        Returns:
            PIL.Image: Rescaled image.
        """
        
        windows = self.get_windows(coords)
        rois = self.calc_roi(windows)
        
        roi = rois[index]
        x, y, w, h = roi
        if isinstance(self.size, int):
            img = img.crop((x, y, (x+w), (y+h)))
            flow = flow.crop((x, y, (x+w), (y+h)))
            w, h = img.size
            if (w <= h and w == self.size) or (h <= w and h == self.size):
                return (img, flow)
            if w < h:
                ow = self.size
                oh = int(self.size * h / w)
                return (img.resize((ow, oh), self.interpolation),
                        flow.resize((ow, oh), self.interpolation))
            else:
                oh = self.size
                ow = int(self.size * w / h)
                return (img.resize((ow, oh), self.interpolation),
                        flow.resize((ow, oh), self.interpolation))
        else:
            return (img.resize(self.size, self.interpolation),
                    flow.resize(self.size, self.interpolation))

        
    def get_windows(self, coords):
        people = coords['people']
        torso = coords['torso']
        windows = []
        for i, p in enumerate(people):
            try:
                if p is not None:
                    window = self.calc_margin(torso, p)
                    windows.append(window)
                else:
                    windows.append((0, 0, 0, 0))
            except:
                print("{} fail to calculate margin".format(i))
                windows.append(None)
        return windows
    
    
    def calc_margin(self, torso, track_window):
        torso = np.array(torso)
        x, y, w, h = track_window

        ## min이나 max가 4, 7일때 (idx:2, 5)일 때
        min_x_idx = np.argmin(torso.T[0])
        max_x_idx = np.argmax(torso.T[0])
        m = 20 # margin
        x, y, w, h = x-m, y-m, w+(m*2), h+(m*2)

        if min_x_idx == 2 or min_x_idx == 5:
            x = x-m
        if max_x_idx == 2 or max_x_idx == 5:
            w = w+m
        #x = 0 if x<0 else x
        #y = 0 if y<0 else y
        return (x, y, w, h)
    
    def calc_roi(self, windows):
        ws = np.array(windows).T[2]
        hs = np.array(windows).T[3]
        max_w, max_w_idx = np.max(ws), np.argmax(ws)
        max_h, max_h_idx = np.max(hs), np.argmax(hs)
        
        rois = []
        for i, track_window in enumerate(windows):
            x, y, w, h = track_window

            if w == 0 and h == 0:
                # bring the first element having w, h
                x, y, w, h = [t for t in windows if t[2] != 0 and t[3] != 0][0]

            x_m = int((max_w-w)/2)
            y_m = int((max_h-h)/2)
            x, y, w, h = x-x_m, y-y_m, w+(x_m)*2, h+(y_m)*2

            rois.append((x,y,w,h))
        return rois
    
    def randomize_parameters(self):
        pass
    
    
class MultiScaleTorsoRandomCrop(CropTorso):
    
    def __init__(self, scales, size, interpolation=Image.BILINEAR):
        super(MultiScaleTorsoRandomCrop, self).__init__(size, interpolation)
        self.scales = scales
            
            
    def __call__(self, img, flow, coords, index):
        """
        Args:
            img (PIL.Image): Image to be scaled.
        Returns:
            PIL.Image: Rescaled image.
        """
        
        windows = self.get_windows(coords)
        rois = self.calc_roi(windows)
        
        roi = rois[index]
        x, y, w, h = roi
        
        if isinstance(self.size, int):
            crop_size_w = int(w * self.scale)
            crop_size_h = int(h * self.scale)

            x -= self.tl_x * np.abs(w-crop_size_w)
            y -= self.tl_y * np.abs(h-crop_size_h)
            x2 = x + crop_size_w
            y2 = y + crop_size_h

            img = img.crop((x, y, x2, y2))
            flow = flow.crop((x, y, x2, y2))
            return (img.resize((self.size, self.size), self.interpolation),
                    flow.resize((self.size, self.size), self.interpolation))
        
        
    def randomize_parameters(self):
        self.scale = self.scales[random.randint(0, len(self.scales)-1)]
        self.tl_x = random.random()
        self.tl_y = random.random()        
    