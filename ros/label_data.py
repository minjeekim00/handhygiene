import numpy as np
import pandas as pd
import os
from tqdm import tqdm
from glob import glob
import shutil

from opts import parse_opts


"""
Seperating raw images into 3 categories; surgery, anesthesia, still(useless images) 
"""
# classes directories
# /images
### /surgery
### /anesthesia
###    /handhygiene (target)
### /still

def get_classes(cls, df):
    classes = []
    for col in df_ex.columns:
        if cls in col:
            classes.append(col)
            
    subclasses = {}
    if not classes:
        return

    for cls in classes: # subclass
        subclasses[cls] = df[cls].iloc[0] 
    return subclasses

            
def sort_frames_into_categories(label_dir):
    cnt = 0
    images = glob(os.path.join(label_dir,'*.jpg'))
    images.sort()
    
    for image in tqdm(images): 
        img_name= image.split('/')[-1].split('.')[0]
        video_id = int(img_name.split('_')[0])
        video_name = str(video_id)+'_'+img_name.split('_')[1]
        frame_num = img_name[-6:] # ex: 000001
        df = df_ex[df_ex['video_id'] == video_id]

        for phase in ['surgery', 'anesthesia', 'still']:
            frames = get_classes(phase, df)
            
            for procedure in sorted(frames.keys(), reverse=True):
                #if frames[procedure] is np.nan:
                if 'float' in str(type(frames[procedure])):
                    continue
                lists = frames[procedure].strip().split(',')
                for el in lists:
                    start = el.strip().split('-')[0]
                    end = el.strip().split('-')[1]

                    if int(frame_num) in range(int(start), int(end)+1):
                        if not os.path.exists(image):
                            continue

                        #print("image: {}, procedure: {}".format(os.path.basename(image), procedure))
                        
                        if '/' not in procedure: ## for class
                            label_path = os.path.join(img_dir, procedure)
                        else : ## for subclass(labeled)
                            label = procedure.split('/')[0]
                            sublabel = procedure.split('/')[1]
                            label_path = os.path.join(img_dir, label+'_label')
                            if not os.path.exists(label_path):
                                os.mkdir(label_path)
                            label_path = os.path.join(label_path, sublabel)
                            
                        if not os.path.exists(label_path):
                            os.mkdir(label_path)    
                        dst_path = os.path.join(label_path, video_name)
                        if not os.path.exists(dst_path):
                            os.mkdir(dst_path)
                        
                        shutil.move(image, dst_path)
                        #cnt += 1
                        break
                        
    return


if __name__ == "__main__":
    
    opt = parse_opts()
    root_dir = opt.root_dir
    img_dir = opt.img_dir
    label_dir = opt.label_dir
    excel_path = opt.excel_path
    df_ex = pd.read_excel(excel_path)

    sort_frames_into_categories(label_dir)