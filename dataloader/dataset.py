import torch
import torch.utils.data as data

import os
import numpy as np
import cv2
#from utils import preprocessing
###### for video processing

def make_dataset(dir, class_to_idx):
    fnames, labels = [], []
    for label in sorted(os.listdir(dir)):
        for fname in os.listdir(os.path.join(dir, label)):
            fnames.append(os.path.join(dir, label, fname))
            labels.append(label)
            
    assert len(labels) == len(fnames)
    print('Number of {} videos: {:d}'.format(dir, len(fnames)))
    targets = labels_to_idx(labels)
    
    return [fnames, targets]

def labels_to_idx(labels):
    
    labels_dict = {label: i for i, label in enumerate(sorted(set(labels)))}
    if len(set(labels)) == 2:
        return np.array([np.eye(2)[int(labels_dict[label])] for label in labels])
    else:
        return np.array([labels_dict[label] for label in labels], dtype=int)

def find_classes(dir):
    """
       returns classes, class_to_idx
    """
    classes = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
    classes.sort()
    class_to_idx = {classes[i]: i for i in range(len(classes))}
    return classes, class_to_idx


class VideoDataset(data.Dataset):
    #def __init__(self, root, transform=None, target_transform=None,
    #             loader=default_loader):
        
    def __init__(self, root, split='train', clip_len='16', preprocess=False):
        
        self.root = root
        self.video_dir = os.path.join(root, 'videos')
        self.image_dir = os.path.join(root, 'images')
        folder = os.path.join(self.image_dir, split)
        
        classes, class_to_idx = find_classes(folder)
        samples = make_dataset(folder, class_to_idx) # [fnames, labels]
        self.samples = samples
        self.clip_len = clip_len
        self.resize_height = 224 # 128
        self.resize_width = 224 # 171
        self.crop_size = 224

        if preprocess:
            self.preprocess()

    def __getitem__(self, index):
        # loading and preprocessing.
        fnames, targets = self.samples[0][index], self.samples[1][index]
        print(fnames, targets)
        buffer = self.load_frames(fnames)
        buffer = self.crop(buffer, int(self.clip_len), self.crop_size)
        buffer = self.normalize(buffer)
        
        #labels = np.array(self.label_array[index])
        #labels = np.expand_dims(np.array(self.label_array[index]), 1)
        return torch.from_numpy(buffer), torch.from_numpy(targets)

    def labels_to_idx(labels):
        return {label: i for i, label in enumerate(sorted(set(labels)))}
    
    def to_one_hot(label):
        to_one_hot = np.eye(2)
        return to_one_hot[int(label)]
        
    def check_preprocess(self):
        # TODO: Check image size in image_dir
        if not os.path.exists(self.image_dir):
            return False
        elif not os.path.exists(os.path.join(self.image_dir, 'train')):
            return False

        for ii, video_class in enumerate(os.listdir(os.path.join(self.image_dir, 'train'))):
            for video in os.listdir(os.path.join(self.image_dir, 'train', video_class)):
                video_name = os.path.join(os.path.join(self.image_dir, 'train', video_class, video),
                                    sorted(os.listdir(os.path.join(self.image_dir, 'train', video_class, video)))[0])
                image = cv2.imread(video_name)
                if np.shape(image)[0] != 128 or np.shape(image)[1] != 171:
                    return False
                else:
                    break

            if ii == 10:
                break

        return True
    
    def load_frames(self, file_dir):
        frames = sorted([os.path.join(file_dir, img) for img in os.listdir(file_dir)])
        frame_count = len(frames)

        buffer = np.empty((frame_count, self.resize_height, self.resize_width, 3), np.dtype('float32'))
        for i, frame_name in enumerate(frames):
            frame = cv2.imread(frame_name)
            frame = cv2.resize(frame, (self.resize_width, self.resize_height))
            frame = np.array(frame).astype(np.float64)
            frame -= np.array([[[90.0, 98.0, 102.0]]])
            # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            buffer[i] = frame

        # convert from [T, H, W, C] format to [C, T, H, W] (what PyTorch uses)
        # T = Time, H = Height, W = Width, C = Channels
        buffer = buffer.transpose((3, 0, 1, 2))

        return buffer

    def crop(self, buffer, clip_len, crop_size):
        # randomly select time index for temporal jittering
        if buffer.shape[1] - clip_len > 0:
            time_index = np.random.randint(buffer.shape[1] - clip_len)
        # randomly select start indices in order to crop the video
        #height_index = np.random.randint(buffer.shape[2] - crop_size)
        #width_index = np.random.randint(buffer.shape[3] - crop_size)

        # crop and jitter the video using indexing. The spatial crop is performed on
        # the entire array, so each frame is cropped in the same location. The temporal
        # jitter takes place via the selection of consecutive frames
            buffer = buffer[:, time_index:time_index + clip_len,:,:]
                     #height_index:height_index + crop_size,
                     #width_index:width_index + crop_size]

        return buffer
    def preprocess(self):
        from sklearn.model_selection import train_test_split
        
        if not os.path.exists(self.image_dir):
            os.mkdir(self.image_dir)
            os.mkdir(os.path.join(self.image_dir, 'train'))
            os.mkdir(os.path.join(self.image_dir, 'val'))
            os.mkdir(os.path.join(self.image_dir, 'test'))

        # Split train/val/test sets
        for file in os.listdir(self.root):
            file_path = os.path.join(self.root, file)
            video_files = [name for name in os.listdir(file_path)]

            train_and_valid, test = train_test_split(video_files, test_size=0.2, random_state=42)
            train, val = train_test_split(train_and_valid, test_size=0.2, random_state=42)

            train_dir = os.path.join(self.image_dir, 'train', file)
            val_dir = os.path.join(self.image_dir, 'val', file)
            test_dir = os.path.join(self.image_dir, 'test', file)

            if not os.path.exists(train_dir):
                os.mkdir(train_dir)
            if not os.path.exists(val_dir):
                os.mkdir(val_dir)
            if not os.path.exists(test_dir):
                os.mkdir(test_dir)

            for video in train:
                self.process_video(video, file, train_dir)

            for video in val:
                self.process_video(video, file, val_dir)

            for video in test:
                self.process_video(video, file, test_dir)

        print('Preprocessing finished.')

    def process_video(self, video, action_name, save_dir):
        # Initialize a VideoCapture object to read video data into a numpy array
        video_filename = video.split('.')[0]
        if not os.path.exists(os.path.join(save_dir, video_filename)):
            os.mkdir(os.path.join(save_dir, video_filename))

        capture = cv2.VideoCapture(os.path.join(self.root, action_name, video))

        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Make sure splited video has at least 16 frames
        EXTRACT_FREQUENCY = 4
        if frame_count // EXTRACT_FREQUENCY <= 16:
            EXTRACT_FREQUENCY -= 1
            if frame_count // EXTRACT_FREQUENCY <= 16:
                EXTRACT_FREQUENCY -= 1
                if frame_count // EXTRACT_FREQUENCY <= 16:
                    EXTRACT_FREQUENCY -= 1

        count = 0
        i = 0
        retaining = True

        while (count < frame_count and retaining):
            retaining, frame = capture.read()
            if frame is None:
                continue

            if count % EXTRACT_FREQUENCY == 0:
                if (frame_height != self.resize_height) or (frame_width != self.resize_width):
                    frame = cv2.resize(frame, (self.resize_width, self.resize_height))
                cv2.imwrite(filename=os.path.join(save_dir, video_filename, '0000{}.jpg'.format(str(i))), img=frame)
                i += 1
            count += 1

        # Release the VideoCapture once it is no longer needed
        capture.release()
        
    def normalize(self, buffer):
        # Normalize the buffer
        # NOTE: Default values of RGB images normalization are used, as precomputed
        # mean and std_dev values (akin to ImageNet) were unavailable for Kinetics. Feel
        # free to push to and edit this section to replace them if found.
        buffer = (buffer - 128) / 128
        return buffer

    def __len__(self):
        return len(self.samples[0]) # fnames



if __name__ == "__main__":
    from torch.utils.data import DataLoader
    train_data = VideoDataset(dataset='ucf101', split='test', clip_len=8, preprocess=False)
    train_loader = DataLoader(train_data, batch_size=100, shuffle=True, num_workers=4)

    for i, sample in enumerate(train_loader):
        inputs = sample[0]
        labels = sample[1]
        print(inputs.size())
        print(labels)

        if i == 1:
            break