from __future__ import print_function
import os, shutil
import argparse

import cv2
import rosbag
from rosbag import Bag
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from tqdm import tqdm
from glob import glob
import roslz4

import pandas as pd
import numpy as np


parser = argparse.ArgumentParser(description="Extract images from bag files")
parser.add_argument("--bag_dir", default='/media/minjee/4970a4b3-9bec-42aa-8022-ddff6e7b8f80/bagfiles/')
parser.add_argument("--img_dir", default= '/media/minjee/4970a4b3-9bec-42aa-8022-ddff6e7b8f80/images/')
parser.add_argument("--excel_path", default= '/data/handhygiene/hh_label.xlsx')
#parser.add_argument("--data_num", type=int)

args = parser.parse_args()

bag_dir = args.bag_dir
img_dir = args.img_dir
excel_path = args.excel_path
df_ex = pd.read_excel(excel_path)
bags = sorted(glob(bag_dir + '*.bag'), key=os.path.getmtime)

def get_last_data_num():
    n_data = df_ex['video_id'].iloc[-1]
    print("starting data number from {}".format(int(n_data)+1))
    
    return int(n_data)+1

def set_camera_info(i, rosbag, bag):
    # get fps, height and width from camera info topic
    fps = 0
    height = 0
    width = 0
    info='/device_0/sensor_1/Color_0/info'
    camera_info='/device_0/sensor_1/Color_0/info/camera_info'
    date = os.path.basename(bag).split('_')[0]
    print(bag)
    
    for _, msg, _ in rosbag.read_messages(topics=[info]):
        fps = msg.fps
        if fps != 0:
            break
        
    for _, msg, _ in rosbag.read_messages(topics=[camera_info]):
        height = msg.height
        width = msg.width
        if height != 0 and width != 0:
            break
    
    print(i, fps, height, width)
    df_ex.at[i, 'fps'] = fps
    df_ex.at[i, 'height'] = height
    df_ex.at[i, 'width'] = width
    df_ex.at[i, 'video_id'] = i+1
    df_ex.at[i, 'date'] = date
    df_ex.at[i, 'video_name'] = str(i+1)+'_'+str(date)
    
    
    return 

def extract_images(add_data_from, rosbag):
    
    #depth_image_data='/device_0/sensor_0/Depth_0/image/data'
    #infrared_image_data='/device_0/sensor_0/Infrared_1/image/data'
    rgb_image_data='/device_0/sensor_1/Color_0/image/data' # we're only dealing with rgb images for now
    
    count = 0
    
    for topic, msg, t in tqdm(rosbag.read_messages(topics=[rgb_image_data])):
        
        if msg is not None:
            desired_encoding = "bgr8" if msg.encoding == "rgb8" else "passthrough" 
            cv_img = bridge.imgmsg_to_cv2(msg, desired_encoding)

            if not os.path.exists(img_dir):
                os.path.mkdir(img_dir)
            
            date = bag.split('/')[-1].split('_')[0]
            target_dir = os.path.join(img_dir, "{}_{}_frames{:06d}.jpg".format(add_data_from, date, count))
            cv2.imwrite(target_dir, cv_img)
            #print("Wrote image {} to {}".format(count, target_dir))
            
        #shutil.move(bag, os.path.join(bag_dir, 'extracted'))
        count += 1
        
    rosbag.close()
    
    return

for i, bag in enumerate(tqdm(bags)):
    rosbag = Bag(bag, "r", allow_unindexed=True)
    bridge = CvBridge()
    
    global add_data_from
    add_data_from = get_last_data_num()
    #set_camera_info(data_num-1, rosbag, bag)
    extract_images(add_data_from, rosbag)
    add_data_from += 1
    
    
    