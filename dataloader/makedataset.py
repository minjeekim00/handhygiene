import os
import json
import numpy as np
import pandas as pd
from torchvision.datasets.folder import IMG_EXTENSIONS
from torchvision.datasets.folder import has_file_allowed_extension

def is_image_file(filename):
    return has_file_allowed_extension(filename, IMG_EXTENSIONS)


def make_hh_dataset(dir, class_to_idx, df, exclusions, cropped):
    """
        fnames: name of directory containing images
        coords: dict containg people, torso coordinates
        labels: class
    """
    #np.random.seed(50)
    fnames, coords, labels = [], [], []
    lists = df['imgpath'].values
    
    for label in os.listdir(os.path.join(dir)):
        for fname in os.listdir(os.path.join(dir, label)):
            try:
                
                #if fname not in lists:
                #    continue
                if any(exc in fname for exc in exclusions):
                    continue

                frames = sorted([img for img in os.listdir(os.path.join(dir, label, fname))])
                frames = [img for img in frames if is_image_file(img)]
                isCropped = True if len(fname.split('_'))>3 else False
                txtfile = os.path.join(dir, label, fname, fname+'.txt')
                with open(txtfile, 'r') as f:
                    item = json.load(f)
                people = item['people']
                npeople = np.array(people).shape[0]
                torso = item['torso']

                target = '_'.join(fname.split('_')[:-1]) if isCropped else fname
                tidxs = df[df['imgpath']==target]['targets'].values # target idx
                if len(tidxs) == 0:  continue # remove data without label
                else: tidxs = tidxs[0]

                #class = df[df['imgpath']==fname]['class'].values[0] # label 
                tidxs = [int(t) for t in tidxs.strip().split(',')]
                nidxs = list(range(npeople))
                nidxs = [int(n) for n in nidxs if n not in tidxs]
                ## appending clean
                for tidx in tidxs:
                    if isCropped:
                        start=int(fname.split('_')[-1])
                    else:
                        start=0
                    end=start+len(frames)

                    if len(frames) != len(people[tidx][start:end]) and not isCropped:
                        print("<{}> {} coords and {} frames / of people {}"
                                  .format(fname, len(people[tidx]), len(frames), tidx))
                        print(people[tidx])
                        continue

                    fnames.append(os.path.join(dir, label, fname))
                    if not cropped:
                        coords.append({'people':people[tidx][start:end], 
                                       'torso':torso[tidx][start:end]})
                    else: ## TODO: change 224 to image shape
                        coords.append({'people': [[0, 0, 224, 224] for i in range(start, end)],
                                       'torso':torso[tidx][start:end]})

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
            except:
                print("exception:", fname)
    
    print('Number of {} people: {:d}'.format(dir, len(fnames)))
    return fnames, coords, labels

def target_dataframe(path='./data/label.csv'):
    df=pd.read_csv(path)
     # target 있는 imgpath만 선별
    df = df[df['targets'].notnull()]
    return df
