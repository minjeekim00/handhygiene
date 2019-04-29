import torch.utils.data as data
from torchvision.datasets import ImageFolder
from PIL import Image

import os
import os.path
import json
import numpy as np
import pandas as pd

from .poseroi import calc_margin
from .poseroi import crop_by_clip

IMG_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', 'webp']


def has_file_allowed_extension(filename, extensions):
    filename_lower = filename.lower()
    return any(filename_lower.endswith(ext) for ext in extensions)

def is_image_file(filename):
    return has_file_allowed_extension(filename, IMG_EXTENSIONS)


#### roi dataset from bounding box
def make_dataset(dir, class_to_idx):
    """
        fnames: name of directory containing images
        coords: dict containg people, torso coordinates
        labels: class
    """
    np.random.seed(50)
    print("making cropped dataset.....")
    fnames, coords, labels = [], [], []
    
    keypoint = '/data/private/minjee-video/handhygiene/data/keypoints.txt'
    df=pd.read_csv('/data/private/minjee-video/handhygiene/data/label.csv')
    
    exclusions = ['38_20190119_frames000643']
    # target 있는 imgpath만 선별
    df = df[df['targets'].notnull()]
    lists = df['imgpath'].values
    
    with open(keypoint, 'r') as file:
        data = json.load(file)
        data = data['coord']
        for fname in os.listdir(os.path.join(dir)):
            if is_image_file(fname):
                continue
            if fname not in lists:
                continue
            if fname in exclusions:
                continue
                
            frames = sorted([img for img in os.listdir(os.path.join(dir, fname))])
            frames = [img for img in frames if is_image_file(img)]
            
            item = [d for d in data if d['imgpath']==fname][0]
            people = item['people']
            torso = item['torso']
            npeople = len(item['people'])
            
            tidxs = df[df['imgpath']==fname]['targets'].values[0] # target idx
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
                
                for i, frame in enumerate(frames):
                    fnames.append(os.path.join(dir, fname, frame))
                    coords.append({'people':people[tidx], 
                                   'torso':torso[tidx]})
                    labels.append('clean')
        
            ## appending notclean 
            if len(nidxs) > 0:
                max = np.random.randint(1, 2+1)
                for nidx in nidxs[:max]:
                    if len(frames) != len(people[nidx]):
                        print("<{}> {} coords and {} frames / of people {}"
                              .format(fname, len(people[nidx]), len(frames), nidx))
                        print(people[nidx])
                        continue
                        
                    for i, frame in enumerate(frames):
                        fnames.append(os.path.join(dir, fname, frame))
                        coords.append({'people':people[nidx], 
                                       'torso':torso[nidx]})
                        labels.append('notclean')
    
    #assert len(labels) == len(fnames)
    print('Number of {} people: {:d}'.format(dir, len(fnames)))
    targets = labels_to_idx(labels)
    
    return [fnames, coords, targets]
    #return [fnames, targets]


def labels_to_idx(labels):
    labels_dict = {label: i for i, label in enumerate(sorted(set(labels)))}
    return np.array([labels_dict[label] for label in labels], dtype=int)


def find_classes(dir):
    """
       returns classes, class_to_idx
    """
    classes = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
    classes.sort()
    class_to_idx = {classes[i]: i for i in range(len(classes))}
    return classes, class_to_idx


def crop_pil_image(coords, idx):
    
    people = coords['people']
    torso = coords['torso']
    try:
        window = people[idx]
        if window is not None:
            window = calc_margin(torso, window)
        else:
            window = (0, 0, 0, 0)
    except:
        print("{} fail to calculate margin".format(idx))
        window = None
        
    return window


def pil_loader(path, coords):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    
    dir = path.replace(os.path.basename(path), '')
    frames = sorted([os.path.join(dir, img) for img in os.listdir(dir)])
    frames = [fname for fname in frames if is_image_file(fname)]
    idx = frames.index(path)
    windows = [crop_pil_image(coords, i) for i in range(len(frames))]
    with open(path, 'rb') as f: 
        img = Image.open(f)
        img = img.convert('RGB')
    
    cropped = crop_by_clip(img, windows, idx)
    return cropped

def accimage_loader(path, coords):
    import accimage
    try:
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path, coords)


def default_loader(path, coords):
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        return accimage_loader(path, coords)
    else:
        return pil_loader(path, coords)
    
    
class ImageDataset(ImageFolder):
    
    def __init__(self, root, transform=None, target_transform=None,
                 loader=default_loader):
        super(ImageFolder, self).__init__(root, loader, IMG_EXTENSIONS,
                                          transform=transform,
                                          target_transform=target_transform)
        classes, class_to_idx = find_classes(root)
        samples = make_dataset(root, class_to_idx)
        self.root = root
        self.classes = classes
        self.class_to_idx = class_to_idx
        self.samples = samples
        
        self.loader = loader
        self.transform = transform
        self.target_transform = target_transform
        
            
    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path = self.samples[0][index]
        coords = self.samples[1][index]
        target = self.samples[2][index]
        sample = self.loader(path, coords)
        
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target
    
    def __getpath__(self, index):
        return self.samples[0][index]
    
    def __len__(self):
        return len(self.samples[0]) # fnames
        