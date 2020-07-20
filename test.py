import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import numpy as np
import os
from tqdm import tqdm
from dataloader.handhygiene import HandHygiene
from model.i3dpt import I3D

from spatial_transforms import (
    Compose, Normalize, Scale, CenterCrop, ToTensor)
from temporal_transforms import (
    MirrorLoopPadding, LoopPadding, TemporalBeginCrop, 
    TemporalRandomCrop, TemporalCenterCrop, TemporalRandomChoice)

from model.i3dpt import I3D, Unit3Dpy
from model.i3d import I3D_binary


rgb_weights_path = 'weights/handhygiene_i3d_rgb_bestiters_1040.pth'
flow_weights_path = 'weights/handhygiene_i3d_flow_bestiters_1040.pth'


def change_key(ordereddict):
    statedict = ordereddict.copy()
    for i, key in enumerate(ordereddict.keys()):
        key, value = statedict.popitem(False)
        old = key
        statedict[key.replace('module.', '') if key == old else key] = value
    return statedict
    
def get_models(num_classes, feature_extract, training_num=0):
    
    i3d_rgb = I3D(num_classes=num_classes, modality='rgb', dropout_prob=0.5)
    statedict = change_key(torch.load(rgb_weights_path))
    i3d_rgb.load_state_dict(statedict)

    i3d_flow = I3D(num_classes=num_classes, modality='flow', dropout_prob=0.5)
    statedict = change_key(torch.load(flow_weights_path))
    i3d_flow.load_state_dict(statedict)
    return i3d_rgb, i3d_flow


def train(models, dataloaders, optimizer, criterion, scheduler, device, num_epochs=50): 
    since = time.time()
    i3d_rgb, i3d_flow = models
    best_model_wts = {'rgb':i3d_rgb.state_dict(), 'flow':i3d_flow.state_dict()}
    best_acc = 0.0
    iterations = {'train': 0, 'val': 0}
    
    for epoch in tqdm(range(num_epochs)):
        
        for phase in ['train', 'val']:
            if phase == 'train':
                scheduler['rgb'].step()
                scheduler['flow'].step()
                i3d_rgb.train()
                i3d_flow.train()
            else:
                i3d_rgb.eval()
                i3d_flow.eval()

            running_loss = 0.0
            running_corrects = {'rgb':0, 'flow':0, 'joint':0}
            
            for i, (samples) in enumerate(dataloaders[phase]):
                iterations[phase] += 1
                rgbs = samples[0] #BCDHW
                flows = samples[1]
                targets = Variable(samples[2].to(device)).float()
                
                ##### rgb model
                optimizer['rgb'].zero_grad()
                rgbs = Variable(rgbs.to(device))
                rgb_out_vars, rgb_out_logits = i3d_rgb(rgbs)
                rgb_preds = torch.round(rgb_out_vars.data)
                
                ##### flow model
                optimizer['flow'].zero_grad()
                flows = Variable(flows.to(device))
                flow_out_vars, flow_out_logits = i3d_flow(flows)
                flow_preds = torch.round(flow_out_vars.data)
                
                with torch.set_grad_enabled(phase == 'train'):
                    ##### joint model
                    out_logit = rgb_out_logits + flow_out_logits
                    out_sigmoid = torch.sigmoid(out_logit)
                    out_preds = torch.round(out_sigmoid.data)
                    out_loss= criterion(out_sigmoid, targets).to(device)
                
                    # backward + optimize only if in training phase
                    if phase == 'train':
                        out_loss.backward()
                        optimizer['rgb'].step()
                        optimizer['flow'].step()
                
                running_loss += out_loss.item() * rgbs.size(0)
                running_corrects['rgb'] += torch.sum(rgb_preds.to(device) == targets.to(device)) 
                running_corrects['flow'] += torch.sum(flow_preds.to(device) == targets.to(device)) 
                running_corrects['joint'] += torch.sum(out_preds.to(device) == targets.to(device)) 
            
            ## for plotting 
            # per epoch
            if phase == 'train':
                train_epoch_loss = running_loss / len(dataloaders[phase].dataset)
                train_epoch_rgb_acc = running_corrects['rgb'].double()  / len(dataloaders[phase].dataset)
                train_epoch_flow_acc = running_corrects['flow'].double()  / len(dataloaders[phase].dataset)
                train_epoch_joint_acc = running_corrects['joint'].double()  / len(dataloaders[phase].dataset)
                    
            else:
                valid_epoch_loss = running_loss / len(dataloaders[phase].dataset)
                valid_epoch_rgb_acc = running_corrects['rgb'].double() / len(dataloaders[phase].dataset)
                valid_epoch_flow_acc = running_corrects['flow'].double() / len(dataloaders[phase].dataset)
                valid_epoch_joint_acc = running_corrects['joint'].double()  / len(dataloaders[phase].dataset)
            
            # deep copy best model
            if phase == 'val' and valid_epoch_joint_acc > best_acc:
                best_acc = valid_epoch_joint_acc
                best_model_rgb_wts = copy.deepcopy(i3d_rgb.state_dict())
                best_model_flow_wts = copy.deepcopy(i3d_flow.state_dict())
                best_iters = iterations['train']
                
        writer.add_scalars('Loss', {'training': train_epoch_loss, 
                                    'validation': valid_epoch_loss}, epoch)
        writer.add_scalars('Accuracy', {'training_rgb': train_epoch_rgb_acc, 
                                        'training_flow': train_epoch_flow_acc,
                                        'training_joint': train_epoch_joint_acc,
                                        'validation_rgb': valid_epoch_rgb_acc,
                                        'validation_flow': valid_epoch_flow_acc,
                                        'validation_joint': valid_epoch_joint_acc}, epoch)
        
        torch.save(i3d_rgb.state_dict(), 
               os.path.join('./weights/{}_{}_epoch_{}.pth'.format('handhygiene', 'i3d_rgb', epoch)))
        torch.save(i3d_flow.state_dict(), 
               os.path.join('./weights/{}_{}_epoch_{}.pth'.format('handhygiene', 'i3d_flow', epoch)))
    
        print('Epoch [{}/{}] train loss: {:.4f} acc: {:.4f} ' 'valid loss: {:.4f} acc: {:.4f}'.format(
                        epoch, num_epochs - 1,
                        train_epoch_loss, train_epoch_joint_acc, 
                        valid_epoch_loss, valid_epoch_joint_acc))
        
    print('Best val Acc: {:4f}'.format(best_acc))

    i3d_rgb.load_state_dict(best_model_rgb_wts)
    i3d_flow.load_state_dict(best_model_flow_wts)
    torch.save(i3d_rgb.state_dict(), 
               os.path.join('./weights/{}_{}_bestiters_{}.pth'.format('handhygiene', 'i3d_rgb', best_iters)))
    torch.save(i3d_flow.state_dict(), 
               os.path.join('./weights/{}_{}_bestiters_{}.pth'.format('handhygiene', 'i3d_flow', best_iters)))
    writer.close()
    return 