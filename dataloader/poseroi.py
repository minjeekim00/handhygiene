import json
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
from glob import glob
from tqdm import tqdm
import cv2

def draw_bbox(imgpath, track_window, margin=False):
    x,y,w,h = track_window
    if margin:
        margin_w = 50
        margin_h = 20
    else:
        margin_w = 0
        margin_h = 0
    
    width = w+margin_w*2
    height = h+margin_h*2
    im = np.array(Image.open(imgpath), dtype=np.uint8)
    # Create figure and axes
    fig,ax = plt.subplots(1, figsize=(6,4))

    # Display the image
    ax.imshow(im)
    ax.axis('off')
    # Create a Rectangle patch
    rect = patches.Rectangle((x-margin_w,y-margin_h),width,height, linewidth=1,edgecolor='r',facecolor='none')
    # Add the patch to the Axes
    ax.add_patch(rect)
    ax.annotate('clean', (x, y), color='cyan', weight='bold', 
                fontsize=10, ha='center', va='center')
    plt.show()

def bb_intersection_over_union(boxA, boxB):
    x11, y11, w1, h1 = boxA
    x21, y21, w2, h2 = boxB
    x12, y12 = x11+w1, y11+h1
    x22, y22 = x21+w2, y21+h2
    xA = np.maximum(x11, np.transpose(x21))
    yA = np.maximum(y11, np.transpose(y21))
    xB = np.minimum(x12, np.transpose(x22))
    yB = np.minimum(y12, np.transpose(y22))
    interArea = np.maximum((xB - xA + 1), 0) * np.maximum((yB - yA + 1), 0)
    boxAArea = (x12 - x11 + 1) * (y12 - y11 + 1)
    boxBArea = (x22 - x21 + 1) * (y22 - y21 + 1)
    iou = interArea / (boxAArea + np.transpose(boxBArea) - interArea)
    # return the intersection over union value
    return iou

    
def get_hand_pose_json(jsondir):
    imgname = os.path.basename(jsondir)
    jsonfiles = sorted(glob(jsondir+'/*.json'))
    people_coords=[]
    imgdir = '/data/openpose/output/image/'
    for index, file in enumerate(jsonfiles[:]):
        with open(file) as f:
            data = json.load(f)
            imgpath = os.path.basename(file)[:-15]+'.jpg'
            image = imgdir+imgname+'/'+os.path.basename(file)[:-14]+'rendered.png'
            people = data.get('people')
            
            for i, p in enumerate(people):
                keypoints = p['pose_keypoints_2d']
                keypoints = np.reshape(keypoints, (25, 3))
                keypoints = keypoints[:,:2] # abandon confidence score
                if keypoints[1].all() == 0 or keypoints[8].all() == 0:
                    continue
                torso = np.array(keypoints[2:8])
                if len([t for t in torso if t.all() == 0])>3: # when missing more then 3 keypoints
                    continue
                x = np.min([c for c in torso.T[0] if c != 0])
                y = np.min([c for c in torso.T[1] if c != 0])
                w = np.max([c for c in torso.T[0] if c != 0])-x
                h = np.max([c for c in torso.T[1] if c != 0])-y
                track_window = (x,y,w,h)
                track_window = [int(t) for t in track_window]
                
                
                if index == 0: # allocate people index
                    people_coords.append([track_window])
                else :#iou로 index에 맞춰서 append
                    ious = [bb_intersection_over_union(x[-1], track_window) for x in people_coords]
                    maxiou = np.max(ious)
                    pidx = np.argmax(ious)
                    if len(people) > len(people_coords) and maxiou < 0.3:
                        length=len(people_coords[0])
                        newpeople = [None for i in range(length-1)] # filling empty coords
                        newpeople.append(track_window)
                        people_coords.append(newpeople)
                        #print("appending to person{}".format(len(people_coords)-1))
                    ious = [bb_intersection_over_union(x[-1], track_window) for x in people_coords]
                    maxiou = np.max(ious)
                    pidx = np.argmax(ious)
                    people_coords[pidx].append(track_window)
                    #print("allocated to person{}".format(pidx))
                draw_bbox(image, track_window)
    coord = {'imgpath': imgpath, 'people': people_coords, 'targets': []}
    coords['coord'].append(coord)
    return 


def calc_margin(torso, track_window):
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


def crop_by_clip(images, coords, idx=None, mode='rgb'):
    """
        rbgs: type: PIL.Image
        coords: (num, (x, y, w, h))
    """
    
    ws = np.array(coords).T[2]
    hs = np.array(coords).T[3]
    
    max_w, max_w_idx = np.max(ws), np.argmax(ws)
    max_h, max_h_idx = np.max(hs), np.argmax(hs)
    
    cropped = []
    for i, track_window in enumerate(coords):
        ## extra margin by max_w, max_h
        x, y, w, h = track_window
        
        if w == 0 and h == 0:
            # bring the first element having w, h
            x, y, w, h = [t for t in coords if t[2] != 0 and t[3] != 0][0]
        
        x_m = int((max_w-w)/2)
        y_m = int((max_h-h)/2)
        x, y, w, h = x-x_m, y-y_m, w+(x_m)*2, h+(y_m)*2
        
        #left = x if x>0 else 0
        #upper = y if y>0 else 0
        
        # when single image
        if not isinstance(images, list) and idx is not None:
            window = (x,y,w,h)
            img = images.crop((x, y, (x+w), (y+h)))
            #img = mean_padding(img, window, (images.size), mode)
            return  img
        
        # for multiple images
        else:
            window = (x,y,w,h)
            img = images[i].crop((x, y, (x+w), (y+h)))
            
            # mean padding
            #img = mean_padding(img, window, (images[i].size), mode)

            # if y < 0, if x > 224, y > 224
            cropped.append(img)
            
    return cropped
        
def mean_padding(img, window, size, mode):
    (x,y,w,h) = window
    shape_w, shape_h = size
    if x < 0:
        img = np.array(img)
        value = np.abs(x)
        means = [np.mean(img[:,value:,c]) for c in range(3)]
        for i in range(3):
            img[:,:value,i]=means[i] if mode == 'rgb' else float(255/2)
            img = Image.fromarray(img)
    
    elif x+w > shape_w:
        img = np.array(img)
        value = x+w-shape_w
        means = [np.mean(img[:,:-1*value,c]) for c in range(3)]
        for i in range(3):
            img[:,-1*value:,i]=means[i] if mode == 'rgb' else float(255/2)
        img = Image.fromarray(img)
    
    # TODOs
    # if y < 0, if x > 224, y > 224
    
    return img


if __name__ == '__main__':
    
    coords = {'coord': []} # roi coords
    for idx, jsondir in enumerate(sorted(glob('/data/openpose/output/json/*'))[:1]):
        if len(os.path.basename(jsondir).split('_'))<4:
            print(idx, jsondir)
            get_hand_pose_json(jsondir)
            
    # coords: 