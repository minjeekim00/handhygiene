import os
import sys
import copy
import json
import pathlib
import time
import random
import easydict

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torch.nn.functional as F
import torch.optim as optim
import torch.backends.cudnn as cudnn

import numpy as np
from glob import glob
from tqdm import tqdm

os.environ["CUDA_VISIBLE_DEVICES"]='1, 2'

from dataloader.handhygiene import HandHygiene
# from dataloader.handhygiene import BalancedBatchSampler
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR
from torch.optim.lr_scheduler import MultiStepLR

from tensorboardX import SummaryWriter
from spatial_transforms import Compose
from spatial_transforms import Normalize
from spatial_transforms import Scale
from spatial_transforms import CenterCrop
from spatial_transforms import RandomHorizontalFlip
from spatial_transforms import RandomAffine
from spatial_transforms import RandomRotation
from spatial_transforms import ColorJitter
from spatial_transforms import ToTensor 
from temporal_transforms import TemporalRandomChoice
from temporal_transforms import TemporalRandomCrop
from temporal_transforms import LoopPadding, MirrorPadding, MirrorLoopPadding
from openpose_transforms import MultiScaleTorsoRandomCrop


from dataloader.handhygiene import get_classes


parser = argparse.ArgumentParser(
        description='Train Hand Hygiene Action Detection'
)
parser.add_argument('--seed', help='A seed for reproducible training', default=100)
parser.add_argument('--mode', help='Classification mode: binary / multi', default='multi')
parser.add_argument('--task', help='Task for action recognition: classification / detection \
                                    If detection, bounding box annotations are required.', default='detection')
parser.add_argument('--model_path_rgb', help='Path for pretrained models', default="model/model_rgb.pth", required=True)
parser.add_argument('--model_path_flow', help='Path for pretrained models', default="model/model_flow.pth", required=True)
parser.add_argument('--label_list_path', help='A list of labels for training', default="./data/annotations/hh_action_list.txt", required=True)
parser.add_argument('--annotation_path', help='Annotation file path', default="./data/annotations/hh_target.csv", required=True)
parser.add_argument('--label_list_path', help='A list of labels for training', default="./data/annotations/hh_action_list.txt", required=True)
parser.add_argument('--as_action', help='To change a label into another one', type=json.loads)
parser.add_argument('--steps', help='A step between video clips per class', \
                        default= {'touching_equipment': 2,
                                     'wearing_gloves': 1,
                                     'rubbing_hands': 1,
                                     'other_action': 6}, type=json.loads)
parser.add_argument('--class_weight', help='Weight per class', type=list)
parser.add_argument('--img_size', help='Input image size for model', default=224, type=int, required=True)
parser.add_argument('--downsample_size', help='Image size when downsampling is needed', default=False)
parser.add_argument('--mean', help='', default=[110.2008, 100.63983, 95.99475], type=list, required=True)
parser.add_argument('--std', help='', default=[58.14765, 56.46975, 55.332195], type=list, required=True)
parser.add_argument('--preproceed_optical_flow', help='', default=False, required=True)
parser.add_argument('--frame_rate', help='', default=None)
parser.add_argument('--fix_fov', help='', default=True)
parser.add_argument('--crop_upper', help='', default=True)
parser.add_argument('--align', help='', default=False)
parser.add_argument('--padding', help='', default=False)
parser.add_argument('--workers', help='', default=4, type=int, required=True)
parser.add_argument('--epochs', help='', default=100, type=int, required=True)
parser.add_argument('--start_epoch', help='', default=0, type=int, required=True)
parser.add_argument('--clip_len', help='', default=16, required=True)
parser.add_argument('--batch_size', help='', default=256, required=True)
parser.add_argument('--lr', help='', default= 1e-3, required=True)
parser.add_argument('--momentum', help='', default=0.9, required=True)
parser.add_argument('--weight_decay', help='', default=1e-5, required=True)
parser.add_argument('--world_size', help='', default=1)
parser.add_argument('--rank', help='', default=1)
parser.add_argument('--gpu', help='', default=None)
parser.add_argument('--dist_backend', help='', default="nccl")
parser.add_argument('--multiprocessing', help='', default=True, required=True)
parser.add_argument('--multiprocessing_distributed', help='', default=False)




if __name__ == '__main__':
    from train import get_models
    from train import train

    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    cudnn.deterministic = True
    cudnn.benchmark = False

    VIDEO_DIR='./data/images_new' #./data/videos

    scales = np.linspace(0.75, 1.1, num=1000)
    center = 1

    openpose_transform = {
        'train':MultiScaleTorsoRandomCrop(scales, args.img_size),
        'val':MultiScaleTorsoRandomCrop(np.linspace(center, center, num=1), 
                                        args.img_size, centercrop=True)
    }

    spatial_transform = {
        'train': Compose([Scale(args.img_size),
                          CenterCrop(args.img_size),
                          RandomHorizontalFlip(),
                          ColorJitter(brightness=0.1),
                          ToTensor(1), 
                          Normalize(args.mean, args.std)
        ]),
        'val': Compose([Scale(args.img_size), 
                        CenterCrop(args.img_size), 
                        ToTensor(1), 
                        Normalize(args.mean, args.std)
        ])}

    temporal_transform = {'train':Compose([ LoopPadding(args.clip_len) ]),
                         'val':LoopPadding(args.clip_len)}

    dataset = {
        'train': HandHygiene(os.path.join(VIDEO_DIR, 'train'),
                             temporal_transform=temporal_transform['train'],
                             openpose_transform=openpose_transform['train'],
                             spatial_transform=spatial_transform['train'],
                             arguments = args
                            ),
        'val': HandHygiene(os.path.join(VIDEO_DIR, 'val'),
                           temporal_transform=temporal_transform['val'],
                           openpose_transform=openpose_transform['val'],
                           spatial_transform=spatial_transform['val'],
                           arguments = args
                          ),
    }

    # create model 
    i3d_rgb, i3d_flow = get_models(len(args.label), True, 170, 
                                   load_pt_weights=True,
                                   rgb_weights_path=args.model_path.rgb,
                                   flow_weights_path=args.model_path.flow)

    if torch.cuda.device_count() > 1:
        i3d_rgb = torch.nn.DataParallel(i3d_rgb).cuda()
        i3d_flow = torch.nn.DataParallel(i3d_flow).cuda()

    # hyperparameters / trainable parameters
    optims={'rgb':None, 'flow':None}
    schedulers = {'rgb':None, 'flow':None}
    feature_extract=True

    def trainable_params(model, mode='rgb'):
        params_to_update = model.parameters()
        print("Params to learn:")
        if feature_extract:
            params_to_update = []
            for name,param in model.named_parameters():
                if param.requires_grad == True:
                    params_to_update.append(param)
                    print("\t",name)
        else:
            for name,param in model.named_parameters():
                if param.requires_grad == True:
                    print("\t",name)

        optims[mode] = torch.optim.SGD(model.parameters(), lr=args.lr, 
                                       momentum=args.momentum,
                                       weight_decay=args.weight_decay)

    trainable_params(i3d_rgb, 'rgb')
    trainable_params(i3d_flow, 'flow')

    schedulers['rgb'] = MultiStepLR(optims['rgb'], milestones=[10], gamma=0.1)
    schedulers['flow'] = MultiStepLR(optims['flow'], milestones=[10], gamma=0.1)

    criterion = F.cross_entropy

    dataloaders = {
            phase: DataLoader(
                dataset[phase], 
                shuffle=True if phase == 'train' else False,
                num_workers=args.workers, 
                #pin_memory=True, 
                batch_size=args.batch_size if phase=='train' else 1,
            )
            for phase in ['train', 'val']
        }

    if args.multiprocessing:
        managers = {'train': mp.Manager(),
                    'val'  : mp.Manager()}

    train((i3d_rgb, i3d_flow), 
          dataloaders, optims, 
          criterion,
          schedulers,
          args.epochs, 
          args.start_epoch,
          managers,
          args)