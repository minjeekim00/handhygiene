import torch
import os
import shutil
import numpy as np
from tqdm import tqdm
from glob import glob
import time


def has_file_allowed_extension(filename, extensions):
    filename_lower = filename.lower()
    return any(filename_lower.endswith(ext) for ext in extensions)

def is_image_file(filename):
    return has_file_allowed_extension(filename, IMG_EXTENSIONS)


def make_dataset(dir, class_to_idx):
    fnames, labels = [], []
    lists = sorted(os.listdir(dir))
    
    for label in sorted(os.listdir(dir)):
        for fname in os.listdir(os.path.join(dir, label)):
            if is_image_file(fname):
                continue
            fnames.append(os.path.join(dir, label, fname))
            labels.append(label)
            
    assert len(labels) == len(fnames)
    print('Number of {} videos: {:d}'.format(dir, len(fnames)))
    targets = labels_to_idx(labels)
    
    return [fnames, targets]

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

IMG_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', 'webp']

def img_path_loader(path):
    """
        return: list of paths of images
    """
    frames = sorted([os.path.join(path, img) for img in os.listdir(path)])
    frames = [fname for fname in frames if is_image_file(fname)]
    return frames


class AugmentDataset():
    
    def __init__(self, root, split='train', clip_len=16):
        self.root =root
        self.image_dir = os.path.join(root, 'images')
        
        folder = os.path.join(self.image_dir, split)
        classes, class_to_idx = find_classes(folder)
        
        self.samples = make_dataset(folder, class_to_idx)
        self.clip_len = clip_len
   
    def __generate_temporal_augmentation__(self, index):
        dirname = self.samples[0][index]
        if len(dirname.split('_')) > 3:
            return
        if '/clean/' in dirname:
            return
        list_frames = img_path_loader(dirname)
        nframes = len(list_frames)
        clip_len = self.clip_len
        
        #print("in {}, {} images found".format(dirname, nframes))
        
        from PIL import Image
        with open(list_frames[0], 'rb') as f:
            img = Image.open(f)
            img = img.convert('RGB')
            shape = np.array(img).shape
            
        np.random.seed(int(time.time()))
        if nframes > clip_len:
            size = nframes-clip_len+1
            if int(size) > int(int(clip_len)/2):
                choices = np.random.choice(size, int(size/(clip_len)), replace=False)
            else:
                choices = np.random.choice(size, int(size/4), replace=False)
            print("dirname: {}, choicable size:{}/{},  choices: {}".format(os.path.basename(dirname), size, nframes, choices))
            for ch in choices:
                if ch == 0:
                    continue
                path_dst = dirname+'_{}'.format(ch-1)
                if os.path.exists(path_dst): 
                    continue
                
                os.mkdir(path_dst)
                path_orig = os.path.join(dirname, list_frames[ch]) # start image
                for i in range(ch, ch+clip_len):
                    shutil.copy(list_frames[i], path_dst)
        return
    
    def __len__(self):
        return len(self.samples[0]) # fnames
   
    
    
if __name__ == "__main__":
    import os
    from tqdm import tqdm
    from dataloader.augmentation import AugmentDataset
    dataset_path = os.getcwd()
    dataset = AugmentDataset(dataset_path, split='train')
    for i, data in enumerate(tqdm(range(dataset.__len__()))):
        dataset.__generate_temporal_augmentation__(i)

        
