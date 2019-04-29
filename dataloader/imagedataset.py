import torch.utils.data as data
from torchvision.datasets import ImageFolder
from PIL import Image

import os
import os.path
import json
import numpy as np
import pandas as pd

IMG_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', 'webp']


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
    #if len(set(labels)) == 2:
    #    return np.array([np.eye(2)[int(labels_dict[label])] for label in labels], dtype=np.float32)
    #else:
    return np.array([labels_dict[label] for label in labels], dtype=int)


def find_classes(dir):
    """
       returns classes, class_to_idx
    """
    classes = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
    classes.sort()
    class_to_idx = {classes[i]: i for i in range(len(classes))}
    return classes, class_to_idx


def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


def accimage_loader(path):
    import accimage
    try:
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)


def default_loader(path):
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        return accimage_loader(path)
    else:
        return pil_loader(path)
    
    
class ImageDataset(ImageFolder):
    
    def __init__(self, root, transform=None, target_transform=None,
                 loader=default_loader):
        super(ImageFolder, self).__init__(root, loader, IMG_EXTENSIONS,
                                          transform=transform,
                                          target_transform=target_transform)
        classes, class_to_idx = find_classes(root)
        self.root = root
        self.classes = classes
        self.class_to_idx = class_to_idx
        self.samples = make_dataset(root, class_to_idx)
        
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
        path, target = self.samples[index]
        sample = self.loader(path)
        
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target
    
    def __getpath__(self, index):
        return self.samples[index]
    
    def __len__(self):
        return len(self.samples[0]) # fnames
        