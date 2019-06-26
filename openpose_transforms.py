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
from datetime import datetime
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
        """ give extra margins to the bbox with min of max hand coord """
        m = 20 # margin
        torso = np.array(torso)
        x, y, w, h = track_window
        
        if len(torso) == 0:
            return (x-m, y-m, w+(m*2), h+(m*2))
        
        x, y, w, h = x-m, y-m, w+(m*2), h+(m*2)
        
        ## min이나 max가 4, 7일때 (idx:2, 5)일 때 마진 더 주기
        min_x_idx = np.argmin(torso.T[0])
        max_x_idx = np.argmax(torso.T[0])
        if min_x_idx == 2 or min_x_idx == 5:
            x = x-m
        if max_x_idx == 2 or max_x_idx == 5:
            w = w+m
        #x = 0 if x<0 else x
        #y = 0 if y<0 else y
        return (x, y, w, h)
    
    
    def calc_roi(self, windows):
        if len(windows)==0:
            print("empty windows")
            
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
            roi = [x,y,w,h]
            #roi = self.align_boundingbox([x,y,w,h])
            rois.append(roi)
        
        ## applying simple moving average
        #for i in list(range(len(rois)+1))[len(rois)::-4][:-1]:
        #    rois = self.moving_average(rois, i).T
        for i in list(range(len(rois)+1))[::-4]:
            rois[i:] = self.moving_average(rois[i:], 4).T
            
        buffer = []
        for roi in rois:
            buffer.append(list(roi))
        return buffer
    
    def moving_average(self, rois, period):
        #buffer = [np.nan] * period
        if period == 0:
            return np.array(rois).T
        buffer = np.zeros((4, len(rois)), dtype=int)
        for n, signal in enumerate(np.array(rois).T):
            for i in range(len(signal)):
                if i < period:
                    buffer[n][i]=signal[i]
                else:
                    buffer[n][i]=int(np.round(signal[i-period:i].mean()))
        return buffer
    
    
    def align_boundingbox(self, roi):
        x, y, w, h = roi
        ratio = w/h
        if ratio > 1:
            y -= int((w-h)/2)
        else:
            x -= int((h-w)/2)
        return [x, y, w, w]
    
    def randomize_parameters(self):
        pass
    
    
class MultiScaleTorsoRandomCrop(CropTorso):
    
    def __init__(self, scales, size, interpolation=Image.BILINEAR, centercrop=False):
        super(MultiScaleTorsoRandomCrop, self).__init__(size, interpolation)
        self.scales = scales
        self.centercrop = centercrop    
            
    def __call__(self, img, flow, coords, index):
        """
        Args:
            img (PIL.Image): Image to be scaled.
        Returns:
            PIL.Image: Rescaled image.
        """
        
        windows = self.get_windows(coords)
        if len(windows)==0:
            return windows
        rois = self.calc_roi(windows)
        
        #print(len(rois), "index:{}".format(index))
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
        random.seed(datetime.now())
        self.scale = self.scales[random.randint(0, len(self.scales)-1)]
        if not self.centercrop:
            self.tl_x = random.random()
            self.tl_y = random.random()
        else:
            self.tl_x = 0.5
            self.tl_y = 0.5
    