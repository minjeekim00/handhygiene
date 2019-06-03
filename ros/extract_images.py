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

from opts import parse_opts


def get_last_data_num():
    """ get last data number from excel file """
    return df_ex['video_id'].iloc[-1] + 1


def set_camera_info(i, rosbag, bag):
    """ get fps, height and width from camera info topic """
    fps = 0
    height = 0
    width = 0
    info='/device_0/sensor_1/Color_0/info'
    camera_info='/device_0/sensor_1/Color_0/info/camera_info'
    date = os.path.basename(bag).split('_')[0]
    print("Setting camera info for {} ....".format(bag))
    print("This takes a while.......")
    
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


def extract_images(num_data, rosbag):
    
    #depth: Depth_0, infrared: Infrared_1
    rgb_image_data='/device_0/sensor_1/Color_0/image/data' # dealing with rgb images only 
    count = 0
    
    # Todo: fixed unindexed bag error
    if rosbag.get_message_count() == 0:
        print("reindexing bag file.....")
        return
        
    for topic, msg, t in tqdm(rosbag.read_messages(topics=[rgb_image_data])):
        if msg is None:
            print('msg is None')
            rosbag.close()
            return
        
        else:
            desired_encoding = "bgr8" if msg.encoding == "rgb8" else "passthrough" 
            cv_img = bridge.imgmsg_to_cv2(msg, desired_encoding)
            date = bag.split('/')[-1].split('_')[0]
            target_dir = os.path.join(img_dir, "{}_{}".format(num_data, date))
            if not os.path.exists(target_dir):
                os.mkdir(target_dir)
            target_path = os.path.join(target_dir, "{}_{}_frames{:06d}.jpg".format(num_data, date, count))
            cv2.imwrite(target_path, cv_img)
            print("Wrote image {} to {}".format(count, target_path))
            
        #shutil.move(bag, os.path.join(os.path.split(bag_dir)[0], 'savedfiles'))
        count += 1
        
    rosbag.close()
    
    return

if __name__ == "__main__":
    
    opt = parse_opts()
    bag_dir = opt.bag_dir
    img_dir = opt.img_dir
    excel_path = opt.excel_path
    df_ex = pd.read_excel(excel_path)
    bags = sorted(glob(os.path.join(bag_dir,'*.bag')), key=os.path.getmtime)
    num_data = get_last_data_num()
    
    for i, bag in enumerate(tqdm(bags)):
        rosbag = Bag(bag, "r", allow_unindexed=True)
        bridge = CvBridge()
        #set_camera_info(num_data-1, rosbag, bag)
        print("Extracting images from {}......".format(bag))
        extract_images(num_data, rosbag)
        num_data += 1