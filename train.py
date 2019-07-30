import os
import time
import copy
import torch
import torch.nn as nn
from torch.autograd import Variable
from model.i3dpt import I3D, Unit3Dpy
from model.i3d import I3D_binary

from tensorboardX import SummaryWriter
from tqdm import tqdm

import logging
model_name = 'i3d'
logpath = os.path.join('./logs/', model_name)
if not os.path.exists(logpath): os.mkdir(logpath)
writer = SummaryWriter(logpath)

rgb_weights_path = 'model/model_rgb.pth'
flow_weights_path = 'model/model_flow.pth'
#rgb_weights_path = 'weights/i3d/handhygiene_i3d_rgb_epoch_99.pth'
#flow_weights_path = 'weights/i3d/handhygiene_i3d_flow_epoch_99.pth'


def set_param_requires_grad(model, feature_extracting, training_num):
    if feature_extracting:
        for i, param in enumerate(model.parameters()):
            if training_num >= i:
                param.requires_grad = False

def change_key(ordereddict):
    statedict = ordereddict.copy()
    
    # cuz of data parallel model
    for i, key in enumerate(ordereddict.keys()):
        key, value = statedict.popitem(False)
        old = key
        statedict[key.replace('module.', '') if key == old else key] = value
    return statedict
        
def get_models(num_classes, feature_extract, training_num=0, load_pt_weights=True):
    
    def modify_last_layer(last_layer, out_channels):
        #last_layer
        conv2 = Unit3Dpy(in_channels=400,
                         out_channels=num_classes, 
                         kernel_size=(1, 1, 1),
                         activation=None, 
                         use_bias=True, use_bn=False)
        branch_0 = torch.nn.Sequential(last_layer, conv2)
        return branch_0
    
    i3d_rgb = I3D(num_classes=400, modality='rgb', dropout_prob=0.5)
    i3d_flow = I3D(num_classes=400, modality='flow', dropout_prob=0.5)
    #i3d_rgb = I3D(num_classes=400, modality='grey', dropout_prob=0.5)### for grey
    
    if load_pt_weights:
        statedict = change_key(torch.load(rgb_weights_path))
        i3d_rgb.load_state_dict(statedict)
    
    set_param_requires_grad(i3d_rgb, feature_extract, training_num)
    i3d_rgb.conv3d_0c_1x1 = modify_last_layer(i3d_rgb.conv3d_0c_1x1, out_channels=num_classes)
    i3d_rgb.softmax = torch.nn.Sigmoid()
    
    if load_pt_weights:
        statedict = change_key(torch.load(flow_weights_path))
        i3d_flow.load_state_dict(statedict)
    
    set_param_requires_grad(i3d_flow, feature_extract, training_num)
    i3d_flow.conv3d_0c_1x1 = modify_last_layer(i3d_flow.conv3d_0c_1x1, out_channels=num_classes)
    i3d_flow.softmax = torch.nn.Sigmoid()
    
    return i3d_rgb, i3d_flow



def train(models, dataloaders, optimizer, criterion, scheduler, device, num_epochs=50, start_epoch=0, flowonly=False):
    since = time.time()
    i3d_rgb, i3d_flow = models
    best_model_wts = {'rgb':i3d_rgb.state_dict(), 'flow':i3d_flow.state_dict(),
                     'joint': {'rgb':i3d_rgb.state_dict(), 'flow':i3d_flow.state_dict()}}
    best_acc = {'rgb':0.0, 'flow':0.0, 'joint':0.0}
    best_iters = {'rgb':0, 'flow':0, 'joint':0}
    #iterations = {'train': 0, 'val': 0}
    iterations = {'train': start_epoch*dataloaders['train'].dataset.__len__(), 
                  'val': start_epoch*dataloaders['val'].dataset.__len__()}
    
    with open(os.path.join(logpath, 'traing_logging.txt'), 'a') as log:
        for epoch in tqdm(range(num_epochs)[start_epoch:]):
            #if epoch > 300:
            #    flowonly=True

            for phase in ['train', 'val']:
                if phase == 'train':
                    #scheduler['rgb'].step()
                    #scheduler['flow'].step()
                    i3d_rgb.train()
                    i3d_flow.train()
                    logging.info('epoch: {}/{}'.format(epoch+1, num_epochs))
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
                    if not flowonly:
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
                        if not flowonly: 
                            out_logit = rgb_out_logits + flow_out_logits
                        else: 
                            out_logit = flow_out_logits
                        out_sigmoid = torch.sigmoid(out_logit)
                        out_preds = torch.round(out_sigmoid.data)
                        out_loss= criterion(out_sigmoid, targets).to(device)

                        # backward + optimize only if in training phase
                        if phase == 'train':
                            out_loss.backward()
                            optimizer['rgb'].step()
                            optimizer['flow'].step()

                    running_loss += out_loss.item() * rgbs.size(0)
                    if not flowonly:
                        running_corrects['rgb'] += torch.sum(rgb_preds.to(device) == targets.to(device)) 
                        running_corrects['joint'] += torch.sum(out_preds.to(device) == targets.to(device))
                    running_corrects['flow'] += torch.sum(flow_preds.to(device) == targets.to(device))


                ## for plotting 
                # per epoch
                if phase == 'train':
                    train_epoch_loss = running_loss / len(dataloaders[phase].dataset)
                    if not flowonly:
                        train_epoch_rgb_acc = running_corrects['rgb'].double()  / len(dataloaders[phase].dataset)
                        train_epoch_joint_acc = running_corrects['joint'].double()  / len(dataloaders[phase].dataset)
                    train_epoch_flow_acc = running_corrects['flow'].double()  / len(dataloaders[phase].dataset)
                else:
                    valid_epoch_loss = running_loss / len(dataloaders[phase].dataset)
                    if not flowonly:
                        valid_epoch_rgb_acc = running_corrects['rgb'].double() / len(dataloaders[phase].dataset)
                        valid_epoch_joint_acc = running_corrects['joint'].double()  / len(dataloaders[phase].dataset)
                    valid_epoch_flow_acc = running_corrects['flow'].double() / len(dataloaders[phase].dataset)

                # deep copy best model
                if phase == 'val':
                    if not flowonly:
                        if valid_epoch_rgb_acc > best_acc['rgb']:
                            best_acc['rgb'] = valid_epoch_rgb_acc
                            best_model_wts['rgb'] = copy.deepcopy(i3d_rgb.state_dict())
                            best_iters['rgb'] = iterations['train']
                        if valid_epoch_joint_acc > best_acc['joint']:
                            best_acc['joint'] = valid_epoch_joint_acc
                            best_model_wts['joint']['rgb'] = copy.deepcopy(i3d_rgb.state_dict())
                            best_model_wts['joint']['flow'] = copy.deepcopy(i3d_flow.state_dict())
                            best_iters['joint'] = iterations['train']

                    if valid_epoch_flow_acc > best_acc['flow']:
                        best_acc['flow'] = valid_epoch_flow_acc
                        best_model_wts['flow'] = copy.deepcopy(i3d_flow.state_dict())
                        best_iters['flow'] = iterations['train']


            writer.add_scalars('Loss', {'training': train_epoch_loss, 
                                        'validation': valid_epoch_loss}, epoch)

            if not flowonly:
                writer.add_scalars('Accuracy', {'training_rgb': train_epoch_rgb_acc, 
                                                'training_flow': train_epoch_flow_acc,
                                                'training_joint': train_epoch_joint_acc,
                                                'validation_rgb': valid_epoch_rgb_acc,
                                                'validation_flow': valid_epoch_flow_acc,
                                                'validation_joint': valid_epoch_joint_acc}, epoch)
                log.write(str(train_epoch_loss)+','
                          +str(valid_epoch_loss)+','
                          +str(train_epoch_rgb_acc.item())+','
                          +str(train_epoch_flow_acc.item())+','
                          +str(train_epoch_joint_acc.item())+','
                          +str(valid_epoch_rgb_acc.item())+','
                          +str(valid_epoch_flow_acc.item())+','
                          +str(valid_epoch_joint_acc.item())+'\n')
            else:
                writer.add_scalars('Accuracy', {'training_flow': train_epoch_flow_acc,
                                                'validation_flow': valid_epoch_flow_acc}, epoch)
            if not flowonly:    
                torch.save(i3d_rgb.state_dict(), 
                       os.path.join('./weights/{}/{}_{}_epoch_{}.pth'.format(model_name, 'handhygiene', 'i3d_rgb', epoch)))
            torch.save(i3d_flow.state_dict(), 
                   os.path.join('./weights/{}/{}_{}_epoch_{}.pth'.format(model_name, 'handhygiene', 'i3d_flow', epoch)))

            print('Epoch [{}/{}] train loss: {:.4f} acc: {:.4f} ' 'valid loss: {:.4f} acc: {:.4f}'.format(
                            epoch, num_epochs - 1,
                            train_epoch_loss, train_epoch_joint_acc if not flowonly else train_epoch_flow_acc, 
                            valid_epoch_loss, valid_epoch_joint_acc if not flowonly else valid_epoch_flow_acc))

        if not flowonly: 
            print('Best Joint val Acc: {:4f}'.format(best_acc['joint']))
            print('Best RGB val Acc: {:4f}'.format(best_acc['rgb']))
        print('Best Flow val Acc: {:4f}'.format(best_acc['flow']))

        ## for joint model
        if not flowonly:
            i3d_rgb.load_state_dict(best_model_wts['joint']['rgb'])
            i3d_flow.load_state_dict(best_model_wts['joint']['flow'])
            torch.save(i3d_rgb.state_dict(), 
                       os.path.join('./weights/{}/{}_{}_bestiters_joint_{}.pth'.format(model_name, 'handhygiene', 'i3d_rgb', best_iters['joint'])))
            torch.save(i3d_flow.state_dict(), 
                       os.path.join('./weights/{}/{}_{}_bestiters_joint_{}.pth'.format(model_name, 'handhygiene', 'i3d_flow', best_iters['joint'])))

        ## for rgb/flow model
        if not flowonly:
            i3d_rgb.load_state_dict(best_model_wts['rgb'])
            torch.save(i3d_rgb.state_dict(), 
                   os.path.join('./weights/{}/{}_{}_bestiters_{}.pth'.format(model_name, 'handhygiene', 'i3d_rgb', best_iters['rgb'])))

        i3d_flow.load_state_dict(best_model_wts['flow'])
        torch.save(i3d_flow.state_dict(), 
                   os.path.join('./weights/{}/{}_{}_bestiters_{}.pth'.format(model_name, 'handhygiene', 'i3d_flow', best_iters['flow'])))

        writer.close()
    return 