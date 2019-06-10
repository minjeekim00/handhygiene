import os
import json
import numpy as np
import pandas as pd
from torchvision.datasets.folder import IMG_EXTENSIONS
from torchvision.datasets.folder import has_file_allowed_extension

def is_image_file(filename):
    return has_file_allowed_extension(filename, IMG_EXTENSIONS)


def make_hh_dataset(dir, class_to_idx, df, data):
    """
        fnames: name of directory containing images
        coords: dict containg people, torso coordinates
        labels: class
    """
    np.random.seed(50)
    fnames, coords, labels = [], [], []
    
    exclusions = ['38_20190119_frames000643', 
                  '40_20190208_frames026493',
                  '34_20190110_frames066161',
                  '34_20190110_frames111213']
    lists = df['imgpath'].values
    
    for label in os.listdir(os.path.join(dir)):
        for fname in os.listdir(os.path.join(dir, label)):
            if is_image_file(fname):
                continue
            if fname not in lists:
                continue
            if fname in exclusions:
                continue

            frames = sorted([os.path.join(dir, img) for img 
                             in os.listdir(os.path.join(dir, label, fname))])
            frames = [img for img in frames if is_image_file(img)]

                
            item = [row for row in data if row['imgpath']==fname][0]
            people = item['people']
            npeople = np.array(people).shape[0]
            torso = item['torso']
            
            tidxs = df[df['imgpath']==fname]['targets'].values[0] # target idx
            if len(tidxs) == 0:
                continue
            #class = df[df['imgpath']==fname]['class'].values[0] # label 
            tidxs = [int(t) for t in tidxs.strip().split(',')]
            nidxs = list(range(npeople))
            nidxs = [int(n) for n in nidxs if n not in tidxs]

            
            ## appending clean
            for tidx in tidxs:
                if len(frames) != len(people[tidx]):
                    print("<{}> {} coords and {} frames / of people {}"
                              .format(fname, len(people[tidx]), len(frames), tidx))
                    print(people[tidx])
                    continue
                fnames.append(os.path.join(dir, label, fname))
                coords.append({'people':people[tidx], 'torso':torso[tidx]})
                labels.append(label)
                
            ## appending notclean 
            #if len(nidxs) > 0:
            #    max = np.random.randint(1, 2+1)
            #    for nidx in nidxs[:max]:
            #        if len(frames) != len(people[nidx]):
            #            print("<{}> {} coords and {} frames / of people {}"
            #                      .format(fname, len(people[nidx]), len(frames), nidx))
            #            print(people[nidx])
            #            continue
                    
            #        fnames.append(os.path.join(dir, label, fname))
            #        coords.append({'people':people[nidx], 'torso':torso[nidx]})
            #        labels.append('notclean')
    
    print('Number of {} people: {:d}'.format(dir, len(fnames)))
    return fnames, coords, labels

def target_dataframe(path='./data/label.csv'):
    df=pd.read_csv(path)
     # target 있는 imgpath만 선별
    df = df[df['targets'].notnull()]
    return df

def get_keypoints(path='./data/keypoints.txt'):
    with open(path, 'r') as file:
        data = json.load(file)
        data = data['coord']
        
        if os.path.exists('./data/keypoints_notclean.txt'):
            
            with open('./data/keypoints_notclean.txt', 'r') as outfile:
                ndata = json.load(outfile)
                ndata = ndata['coord']

                for item in ndata:
                    data.append(item)

        return data