import numpy as np
import pandas as pd
import os
from tqdm import tqdm
from glob import glob
import shutil


def move_target_frames(label_dir):
    labels = glob(os.path.join(label_dir, '*'))
    for label in labels:
        clss = os.path.split(label)[1]
        print("label: {}".format(clss))
        vnames = glob(os.path.join(label, '*'))
        for vname in vnames:
            vdirname = os.path.split(vname)[1]
            if len(vdirname.split('_'))>2:
                continue
            images = sorted(glob(os.path.join(vname, '*.jpg')))

            column = [c for c in df_ex.columns if clss in c]
            li = df_ex[df_ex['video_name']==vdirname][column[-1]].values[0]
            if 'float' in str(type(li)):
                continue
            li = li.strip().split(',')
            print(vdirname)

            for el in li:
                start, end = el.strip().split('-')
                dirname = vname + '_frames' + str(start).zfill(6)
                if not os.path.exists(dirname):
                    os.mkdir(dirname)
                else: 
                    continue
                for i in range(int(start), int(end)+1):
                    imgname = vdirname+'_frames'+str(i).zfill(6)+'.jpg'
                    if os.path.exists(os.path.join(vname,imgname)):
                        shutil.move(os.path.join(vname,imgname), dirname)
                    else:
                        continue
                os.system("rmdir {}".format(vname))
    return

if __name__ == "__main__":
    
    opt = parse_opts()
    label_dir = opt.label_dir
    excel_path = opt.excel_path
    df_ex = pd.read_excel(excel_path)

    move_target_frames(label_dir)