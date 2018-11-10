from __future__ import print_function
import os
import argparse

import cv2
import rosbag
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from tqdm import tqdm
from glob import glob
import roslz4


parser = argparse.ArgumentParser(description="Extract images from bag files")
parser.add_argument("--bag_dir")
parser.add_argument("--img_dir")
parser.add_argument("--data_num", default=1, type=int)

args = parser.parse_args()

bag_dir = args.bag_dir
img_dir = args.img_dir
data_num = args.data_num

def extract_images(bag_dir, img_dir):
    bags = sorted(glob(bag_dir + '*.bag'), key=os.path.getmtime)
    for bag in tqdm(bags):
        date = bag.split('/')[-1].split('_')[0]
        bag = rosbag.Bag(bag, "r", allow_unindexed=True)
        bridge = CvBridge()
        count = 0
        rgb_img_topic = '/device_0/sensor_1/Color_0/image/data'

        for topic, msg, t in bag.read_messages(topics=[rgb_img_topic]):
            if msg is not None:
                cv_img = bridge.imgmsg_to_cv2(msg, "bgr8")

                if not os.path.exists(img_dir):
                    os.mkdir(img_dir)
                target_dir = os.path.join(img_dir, "{}_{}_frames{:06d}.jpg".format(data_num, date, count))
                cv2.imwrite(target_dir, cv_img)
                print("Wrote image {} to {}".format(count, target_dir))
            count += 1
        bag.close()
        
        
extract_images(bag_dir, img_dir)