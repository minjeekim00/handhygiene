import torch
import torch.utils.data as data
import torch.nn.functional as F

import os
import numpy as np
from PIL import Image # to use torchivision transforms
import cv2 # to do flow preprocessing 

from tqdm import tqdm
from glob import glob

import json
import pandas as pd
from .poseroi import calc_margin
from .poseroi import crop_by_clip

def has_file_allowed_extension(filename, extensions):
    filename_lower = filename.lower()
    return any(filename_lower.endswith(ext) for ext in extensions)

def is_image_file(filename):
    return has_file_allowed_extension(filename, IMG_EXTENSIONS)


def make_dataset(dir, class_to_idx):
    fnames, labels = [], []
    lists = sorted(os.listdir(dir))
    
    exclusions = ['5_20181119_frames029980',
     '8_20181122_frames001281',
     '18_20181204_frames006385',
     '38_20190119_frames000270',
     '38_20190119_frames000382',
     '38_20190119_frames000643',
     '38_20190119_frames000905',
     '38_20190119_frames001529',
     '38_20190119_frames001600',
     '38_20190119_frames001633',
     '38_20190119_frames001746',
     '38_20190119_frames002412',
     '38_20190119_frames002615',
     '38_20190119_frames002913',
     '38_20190119_frames003847',
     '38_20190119_frames004411',
     '38_20190119_frames005471',
     '38_20190119_frames006438',
     '38_20190119_frames006587']
    
    for label in sorted(os.listdir(dir)):
        for fname in os.listdir(os.path.join(dir, label)):
            if is_image_file(fname):
                continue
            if fname in exclusions:
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


def default_loader(dir):
    
    rgbs = pil_frame_loader(dir)
    flows = pil_flow_loader(dir)
    return rgbs, flows
            
    
def pil_frame_loader(path):
    """
        return: list of PIL Images
    """
    frames = sorted([os.path.join(path, img) for img in os.listdir(path)])
    
    buffer = []
    for i, fname in enumerate(frames):
        if not is_image_file(fname):
            continue
        with open(fname, 'rb') as f:
            img = Image.open(f)
            img = img.convert('RGB')
            buffer.append(img)
            
    #buffer = np.asarray(buffer, dtype=np.float32)       
    return buffer

def pil_flow_loader(dir):
    """
        return: list of PIL Images
    """
    flow = get_flow(dir)
   
    buffer = []
    for i, flw in enumerate(flow):
        shape = flw.shape
        # to make extra 3 channel to use torchvision transform
        tmp = np.empty((shape[0], shape[1], 1)).astype(np.uint8) 
        img = np.dstack((flw.astype(np.uint8), tmp))
        img = Image.fromarray(img)
        buffer.append(img)
        
    return buffer

############################ for flow processing
    
def compute_TVL1(prev, curr, bound=15):
    """Compute the TV-L1 optical flow."""
    TVL1 = cv2.DualTVL1OpticalFlow_create()
    flow = TVL1.calc(prev, curr, None)
    assert flow.dtype == np.float32

    flow = (flow + bound) * (255.0 / (2*bound))
    flow = np.round(flow).astype(int)
    flow[flow >= 255] = 255
    flow[flow <= 0] = 0
    
    return flow
   
def get_flow(dir):
    """
        return: return: (1, num_frames, H, W, 2) shape of array of .npy
        ##return: (1, num_frames, 224, 224, 2) shape of array of .npy
    """
    basename = os.path.basename(dir)
    if len(basename.split('_')) > 3: # when temporal sampling
        start = int(basename.split('_')[-1])
        currbasename = basename.rsplit('_', 1)[0]
        currdir = dir.rsplit('/', 1)[0]
        flow_dir = os.path.join(currdir, currbasename, '{}.npy'.format(currbasename))
        if os.path.exists(flow_dir):
            flows = np.load(flow_dir)
            return flows[start:start+16] ## clip_len
        #else:
            ## TODO: when base npy not exists
      
    flow_dir = os.path.join(dir,'{}.npy'.format(basename))
    if os.path.exists(flow_dir):
        return np.load(flow_dir)
    
    print("processing optical flows..... this will take for a while....")
    frames = glob(os.path.join(dir, '*.jpg'))
    frames.sort()
    
    flow = []
    prev = cv2.imread(frames[0])
    shape = prev.shape
    #prev = cv2.resize(prev, (224, 224))
    prev = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
    for i, frame_curr in enumerate(tqdm(frames)):
        curr = cv2.imread(frame_curr)
        #curr = cv2.resize(curr, (640, shape[1]))
        curr = cv2.cvtColor(curr, cv2.COLOR_BGR2GRAY)
        tmp_flow = compute_TVL1(prev, curr)
        flow.append(tmp_flow)
        prev = curr
    
    np.save(flow_dir, flow) 
    return flow




#########################################   
IMG_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', 'webp']

class VideoDataset(data.Dataset):
        
    def __init__(self, root, split='train', clip_len=16, transform=None, preprocess=False, 
                 use_keypoints=False, loader=default_loader):

        self.root = root
        self.loader = loader
        self.video_dir = os.path.join(root, 'videos')
        self.image_dir = os.path.join(root, 'images')
        folder = os.path.join(self.image_dir, split)
        
        classes, class_to_idx = find_classes(folder)
        self.classes = classes
        self.samples = make_dataset(folder, class_to_idx) # [fnames, labels]
        self.transform = transform
        self.clip_len = clip_len
        self.use_keypoints = use_keypoints

        if preprocess:
            self.preprocess()

    def __getitem__(self, index):
        # loading and preprocessing.
        """
            TODO: clean up batch size ordering
            sampling 16 frames
        """
        fnames, targets = self.samples[0][index], self.samples[1][index]
        frames, flows = self.loader(fnames)
        
        frames, flows = self.crop_roi((frames, flows), index)
        
        if self.transform: ## applying torchvision transform
            _frames = []
            _flows = []
            for frame in frames: ## for RGB
                frame = self.transform(frame)
                _frames.append(frame)
            for flow in flows:
                ### TODO: flow should be from [-20, 20] to [-1, 1]
                #[0, 255] to [-1, 1] for easy implement (at the moment)
                flow = self.transform(flow) 
                _flows.append(flow[:-1,:,:]) # exclude temp channel 3
            frames = _frames
            flows = _flows
       
        # target transform
        #if len(targets) != 2:
        targets = torch.tensor(targets).unsqueeze(0)
        
        ## temporal transform
        frames, flows = self.temporal_transform((frames, flows), index)
        return frames, flows, targets
    
    def crop_roi(self, streams, index):
        
        path_keyp = '/data/private/minjee-video/handhygiene/data/keypoints.txt'
        df_keyp=pd.read_csv('/data/private/minjee-video/handhygiene/data/keypoints.csv')
        imgpath = self.__getpath__(index)
        
        rgbs = streams[0]
        flows = streams[1]
        
        with open(path_keyp, 'r') as file:
            data = json.load(file)
            data = data['coord']
            for i, row in enumerate(df_keyp.values):
                targets = row[2]
                if targets is np.nan: continue
                idx = int(targets.split(',')[0])
                imgname = data[i]['imgpath']
                people = data[i]['people']
                torso = data[i]['torso']
                
                if imgname == os.path.basename(imgpath):
                    target_coord = people[idx]
                    target_torso = torso[idx]
                    crop_coords = []
                    
                    for j, rgb in enumerate(rgbs[:]):
                        
                        try:
                            target_window = (target_coord[j])
                            if target_window is not None:
                                target_window = calc_margin(torso, target_window)
                                crop_coords.append(target_window)
                            else:
                                target_window = (0, 0, 0, 0)
                                crop_coords.append(target_window)
                        except:
                            print(imgpath)
                            continue
                    cropped = crop_by_clip((rgbs, flows), crop_coords, imgpath)
        return cropped
    
    def to_one_hot(self, label):
        to_one_hot = np.eye(2)
        return to_one_hot[int(label)]
    
    def temporal_transform(self, streams, index):
        """
            all clip length resize to 16
        """
        frames, flows = streams
        clip_len = self.clip_len
        nframes = len(frames)
        
        if nframes == clip_len:
            frames = torch.stack(frames).transpose(0, 1) # DCHW -> CDHW
            flows = torch.stack(flows).transpose(0, 1)
            #print(frames.shape, flows.shape, index)
            return (frames, flows)
        #elif nframes < clip_len:
        #    return self.clip_looping(streams, index)
        
        elif nframes > clip_len:
            return self.clip_sampling(streams, index)
        
        else:
            #frames = self.clip_speed_changer(frames)
            #flows = self.clip_speed_changer(flows)
            frames = torch.stack(frames).transpose(0, 1) # DCHW -> CDHW
            flows = torch.stack(flows).transpose(0, 1)
            return (frames, flows)
    
    def clip_looping(self, streams, index):
        """
            Loop a clip as many times as necessary to satisfy input size
            input shape: DxCxHxW
            return shape: CxDxHxW
        """
        frames, flows = streams
        clip_len = self.clip_len
        nframes = len(frames)
        
        niters = int(clip_len/nframes)+1
        frames = frames*niters
        frames = frames[:clip_len]
        flows = flows*niters
        flows = flows[:clip_len]
        frames = torch.stack(frames).transpose(0, 1)
        flows = torch.stack(flows).transpose(0, 1)
        
        return (frames, flows)
        
    def clip_speed_changer(self, images):
        """
            Interpolate clip size with length of `self.clip_len`
            ex) 25 --> 16
            input shape: DxCxHxW
            return shape: CxDxHxW
        """
        ## TODO: tensor ordering
        images = torch.stack(images)
        shape = images.shape # DCHW
        images = images.transpose(0, 1).unsqueeze(0) #DCHW --> BCDHW
        #`mini-batch x channels x [optional depth] x [optional height] x width`.
        images = F.interpolate(images, size=(self.clip_len, 224, 224), 
                               mode='trilinear', align_corners=True)
        images = images.view(shape[1], self.clip_len, shape[2], shape[3]) #squeeze
        return images #CDHW
    
    def clip_sampling(self, streams, index):
        """
            Sample clip with random start number.
            The length of sampled clip should be same with `self.clip_len`
            input shape: DxCxHxW
            return shape: CxDxHxW
        """
        frames, flows = streams
        clip_len = self.clip_len
        start = 0
        if len(frames) != len(flows):
            print(self.__getpath__(index))
            return print("number of frames {} and flows {} are different.".format(len(frames), len(flows)))
        
        nframes = len(frames)
        if nframes > clip_len:
            size = nframes-clip_len+1
            start = np.random.choice(size, 1)[0]
        elif nframes < clip_len: # drop a clip when its length is less than 16
            print(self.__getpath__(index))
            return print("minimum {} frames are needed to process".format(clip_len))
        
        frames = torch.stack(frames[start:start+clip_len]).transpose(0, 1)
        flows = torch.stack(flows[start:start+clip_len]).transpose(0, 1)
        return (frames, flows)
    
    def clip_add_backwards(self, streams, index):
        return
    
    def check_preprocess(self):
        # TODO: Check image size in image_dir
        if not os.path.exists(self.image_dir):
            return False
        elif not os.path.exists(os.path.join(self.image_dir, 'train')):
            return False
        return True
    
    
    def preprocess(self):
        from sklearn.model_selection import train_test_split
        #### TODO: split train test 
        print('Preprocessing finished.')

    
    def __len__(self):
        return len(self.samples[0]) # fnames
    
    def __getpath__(self, index):
        return self.samples[0][index]


if __name__ == "__main__":
    from torch.utils.data import DataLoader
    
    dataset_path = os.getcwd()
    train_dataset = VideoDataset(dataset_path, split='train', transform=transforms)
    train_loader = DataLoader(dataset['train'], batch_size=2, shuffle=True, num_workers=4)

    for i, sample in enumerate(train_loader):
        rgbs = sample[0]
        flows = sample[1]
        labels = sample[2]
        print(rgbs.size())
        print(flows.size())
        print(labels)

        if i == 1:
            break