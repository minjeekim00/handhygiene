### https://github.com/Rhythmblue/i3d_finetune/issues/2

import os
import numpy as np
import cv2 # to do flow preprocessing
from glob import glob
from tqdm import tqdm


def compute_TVL1(prev, curr, bound=15):
    """Compute the TV-L1 optical flow."""
    TVL1 = cv2.DualTVL1OpticalFlow_create()
    flow = TVL1.calc(prev, curr, None)
    assert flow.dtype == np.float32

    flow = (flow + bound) * (255.0 / (2*bound))
    flow = np.round(flow).astype(int)
    flow[flow >= 255] = 255
    flow[flow <= 0] = 0
    
    return flow

def cal_for_frames(dir):
    frames = glob(os.path.join(dir, '*.jpg'))
    frames.sort()
    
    flow = []
    prev = cv2.imread(frames[0])
    shape = prev.shape
    #prev = cv2.resize(prev, (224, 224))
    prev = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
    for i, frame_curr in enumerate(tqdm(frames)):
        curr = cv2.imread(frame_curr)
        #curr = cv2.resize(curr, (640, shape[1]))
        curr = cv2.cvtColor(curr, cv2.COLOR_BGR2GRAY)
        tmp_flow = compute_TVL1(prev, curr)
        flow.append(tmp_flow)
        prev = curr
    
    np.save(flow_dir, flow)
    return flow
