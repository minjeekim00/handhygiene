import os
import time
import copy
import pathlib
import torch
import torch.nn as nn
from torch.autograd import Variable
from model.i3dpt import I3D, Unit3Dpy

from tensorboardX import SummaryWriter
from tqdm import tqdm


import logging

model_name = 'i3d'
logpath = os.path.join('./weights/', model_name, time.strftime("%Y%m%d"))
pathlib.Path(logpath).mkdir(parents=True, exist_ok=True)
writer = SummaryWriter(logpath)

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
        
    
def get_in_channels(ordereddict, custom_key='conv3d_0c_1x1.1.conv3d.'):
    statedict = ordereddict.copy()
    
    # cuz of data parallel model
    for i, key in enumerate(ordereddict.keys()):
        key, value = statedict.popitem(False)
        old = key
        
        if custom_key in key:
            return value.shape[0]
        
        
def get_models(num_classes, feature_extract, training_num=0, load_pt_weights=True,
              rgb_weights_path='model/model_rgb.pth',
              flow_weights_path = 'model/model_flow.pth'):
    
    if load_pt_weights:
        print("loading rgb weight from {} ....".format(rgb_weights_path))
        print("loading flow weight from {} ....".format(flow_weights_path))
        
    def modify_last_layer(out_channels):
        #last_layer
        conv2 = Unit3Dpy(in_channels=400,
                         out_channels=out_channels, 
                         kernel_size=(1, 1, 1),
                         activation=None, 
                         use_bias=True, use_bn=False)
        #branch_0 = torch.nn.Sequential(last_layer, conv2)
        #return branch_0
        return conv2
    
    i3d_rgb = I3D(num_classes=400, modality='rgb', dropout_prob=0.5)
    i3d_flow = I3D(num_classes=400, modality='flow', dropout_prob=0.5)
    #i3d_rgb = I3D(num_classes=400, modality='grey', dropout_prob=0.5)
    
    if load_pt_weights:
        
        if rgb_weights_path == 'model/model_rgb.pth': # default model
            i3d_rgb.load_state_dict(change_key(torch.load(rgb_weights_path)))
            set_param_requires_grad(i3d_rgb, feature_extract, training_num)
            i3d_rgb.conv3d_0c_1x1 = torch.nn.Sequential(i3d_rgb.conv3d_0c_1x1, modify_last_layer(num_classes))
        else:
            # change channel number
            in_channels = get_in_channels(torch.load(rgb_weights_path))
            i3d_rgb.conv3d_0c_1x1 = torch.nn.Sequential(i3d_rgb.conv3d_0c_1x1, modify_last_layer(in_channels))
            i3d_rgb.load_state_dict(change_key(torch.load(rgb_weights_path)))
            
            set_param_requires_grad(i3d_rgb, feature_extract, training_num)
            i3d_rgb.conv3d_0c_1x1 = torch.nn.Sequential(i3d_rgb.conv3d_0c_1x1[0], modify_last_layer(num_classes))
    
    if load_pt_weights:
        
        if flow_weights_path == 'model/model_flow.pth': # default model
            i3d_flow.load_state_dict(change_key(torch.load(flow_weights_path)))
            set_param_requires_grad(i3d_flow, feature_extract, training_num)
            i3d_flow.conv3d_0c_1x1 = torch.nn.Sequential(i3d_flow.conv3d_0c_1x1, modify_last_layer(num_classes))
        else:
            # change channel number
            in_channels = get_in_channels(torch.load(flow_weights_path))
            i3d_flow.conv3d_0c_1x1 = torch.nn.Sequential(i3d_flow.conv3d_0c_1x1, modify_last_layer(in_channels))
            i3d_flow.load_state_dict(change_key(torch.load(flow_weights_path)))
            
            set_param_requires_grad(i3d_flow, feature_extract, training_num)
            i3d_flow.conv3d_0c_1x1 = torch.nn.Sequential(i3d_flow.conv3d_0c_1x1[0], modify_last_layer(num_classes))
            
    #i3d_rgb.softmax = torch.nn.Sigmoid()
    #i3d_flow.softmax = torch.nn.Sigmoid()
    
    return i3d_rgb, i3d_flow


def train(models, dataloaders, optimizer, criterion, scheduler, num_epochs=50, start_epoch=0, manager=None):
    
    since = time.time()
    i3d_rgb, i3d_flow = models
    best_model_wts = {'rgb':i3d_rgb.state_dict(), 'flow':i3d_flow.state_dict(),
                     'joint': {'rgb':i3d_rgb.state_dict(), 'flow':i3d_flow.state_dict()}}
    best_acc = {'rgb':0.0, 'flow':0.0, 'joint':0.0}
    best_iters = {'rgb':0, 'flow':0, 'joint':0}
    iterations = {'train': start_epoch*dataloaders['train'].dataset.__len__(), 
                  'val': start_epoch*dataloaders['val'].dataset.__len__()}
    
    
    for epoch in tqdm(range(num_epochs)[start_epoch:]):

        for phase in ['train', 'val']:
            if epoch == start_epoch:
                
                if manager is not None:
                    #dataloaders[phase].dataset.video_clips.cache_all()
                    dataloaders[phase].dataset.video_clips.set_shared_manager(manager)
                    dataloaders[phase].dataset.optflow_clips.set_shared_manager(manager)
                    dataloaders[phase].dataset.video_clips.cache_video_all()
                    dataloaders[phase].dataset.optflow_clips.cache_video_all()
                
            if phase == 'train':
                i3d_rgb.train()
                i3d_flow.train()
            else:
                i3d_rgb.eval()
                i3d_flow.eval()

            running_loss = 0.0
            running_corrects = {'rgb':0, 'flow':0, 'joint':0}
            
            ''' for class weight '''
            num_per_class = dataloaders[phase].dataset._num_clips_per_class()
            classes = dataloaders[phase].dataset.classes
            total_num = sum([num_per_class[key] for key in num_per_class])
            #weight_per_class = {cls: 1-(num_per_class[cls]/total_num) for cls in classes}
            #weight_per_class = torch.tensor([weight_per_class[cls] for cls in classes]).cuda()
            
            weight_per_class = torch.tensor([1.25, 1.0, 1.0]).cuda()
            
            for i, (samples) in enumerate(dataloaders[phase]):
                iterations[phase] += 1
                rgbs = samples[0] #BCDHW
                flows = samples[1]
                targets = Variable(samples[2].cuda()) #.float()

                ''' rgb model '''
                optimizer['rgb'].zero_grad()
                rgbs = Variable(rgbs.cuda())
                rgb_out_vars, rgb_out_logits = i3d_rgb(rgbs)
                #rgb_preds = torch.round(rgb_out_vars.data)
                rgb_preds = torch.tensor([torch.argmax(var) 
                                          for var in rgb_out_vars.data])
                ''' flow model '''
                optimizer['flow'].zero_grad()
                flows = Variable(flows.cuda())
                flow_out_vars, flow_out_logits = i3d_flow(flows)
                #flow_preds = torch.round(flow_out_vars.data)
                flow_preds = torch.tensor([torch.argmax(var) 
                                          for var in flow_out_vars.data])

                with torch.set_grad_enabled(phase == 'train'):
                    out_logit = (rgb_out_logits + flow_out_logits)/2
                    #out_sigmoid = torch.sigmoid(out_logit)
                    #out_preds = torch.round(out_sigmoid.data)
                    #out_loss= criterion(out_sigmoid, targets).cuda()
                    
                    out_softmax = torch.softmax(out_logit, 1)
                    out_preds = torch.tensor([torch.argmax(var) 
                                  for var in out_softmax.data])
                    out_loss = criterion(out_softmax, targets
                                            , weight = weight_per_class
                                            ).cuda()

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        out_loss.backward()
                        optimizer['rgb'].step()
                        optimizer['flow'].step()
            
                running_loss += out_loss.item() * rgbs.size(0)
                running_corrects['rgb'] += torch.sum(rgb_preds.cuda() == targets.cuda())
                running_corrects['joint'] += torch.sum(out_preds.cuda() == targets.cuda())
                running_corrects['flow'] += torch.sum(flow_preds.cuda() == targets.cuda())

            ## for plotting 
            # per epoch
            if phase == 'train':
                train_epoch_loss = running_loss / len(dataloaders[phase].dataset)
                train_epoch_rgb_acc = running_corrects['rgb'].double()  / len(dataloaders[phase].dataset)
                train_epoch_joint_acc = running_corrects['joint'].double()  / len(dataloaders[phase].dataset)
                train_epoch_flow_acc = running_corrects['flow'].double()  / len(dataloaders[phase].dataset)
            else:
                valid_epoch_loss = running_loss / len(dataloaders[phase].dataset)
                valid_epoch_rgb_acc = running_corrects['rgb'].double() / len(dataloaders[phase].dataset)
                valid_epoch_joint_acc = running_corrects['joint'].double()  / len(dataloaders[phase].dataset)
                valid_epoch_flow_acc = running_corrects['flow'].double() / len(dataloaders[phase].dataset)
            
            
            if phase == 'train':
                # if pytorch version > 1.1.0
                if scheduler['rgb'] is not None:
#                     scheduler['rgb'].step()
                    writer.add_scalars('learning_rate', 
                                      {'rgb': scheduler['rgb'].get_last_lr()[0]}, epoch)
                if scheduler['flow'] is not None:
#                     scheduler['flow'].step()
                    writer.add_scalars('learning_rate', 
                                      {'flow': scheduler['flow'].get_last_lr()[0]}, epoch)

            # deep copy best model
            if phase == 'val':
                if valid_epoch_rgb_acc > best_acc['rgb']:
                    best_acc['rgb'] = valid_epoch_rgb_acc
                    best_model_wts['rgb'] = copy.deepcopy(i3d_rgb.state_dict())
                    best_iters['rgb'] = iterations['train']
                if valid_epoch_flow_acc > best_acc['flow']:
                    best_acc['flow'] = valid_epoch_flow_acc
                    best_model_wts['flow'] = copy.deepcopy(i3d_flow.state_dict())
                    best_iters['flow'] = iterations['train']
                if valid_epoch_joint_acc > best_acc['joint']:
                    best_acc['joint'] = valid_epoch_joint_acc
                    best_model_wts['joint']['rgb'] = copy.deepcopy(i3d_rgb.state_dict())
                    best_model_wts['joint']['flow'] = copy.deepcopy(i3d_flow.state_dict())
                    best_iters['joint'] = iterations['train']

        writer.add_scalars('Loss', {'training': train_epoch_loss, 
                                    'validation': valid_epoch_loss}, epoch)

        writer.add_scalars('Accuracy', {'training_rgb': train_epoch_rgb_acc, 
                                            'training_flow': train_epoch_flow_acc,
                                            'training_joint': train_epoch_joint_acc,
                                            'validation_rgb': valid_epoch_rgb_acc,
                                            'validation_flow': valid_epoch_flow_acc,
                                            'validation_joint': valid_epoch_joint_acc}, epoch)

        torch.save(i3d_rgb.state_dict(), 
                   os.path.join(logpath, '{}_{}_epoch_{}.pth'.format('handhygiene', 'i3d_rgb', epoch)))
        torch.save(i3d_flow.state_dict(), 
               os.path.join(logpath, '{}_{}_epoch_{}.pth'.format('handhygiene', 'i3d_flow', epoch)))

        print('Epoch [{}/{}] train loss: {:.4f} acc: {:.4f} ' 'valid loss: {:.4f} acc: {:.4f}'.format(epoch, num_epochs - 1, train_epoch_loss, train_epoch_joint_acc, valid_epoch_loss, valid_epoch_joint_acc))

    manager.shutdown()
    print('Best Joint val Acc: {:4f}'.format(best_acc['joint']))
    print('Best RGB val Acc: {:4f}'.format(best_acc['rgb']))
    print('Best Flow val Acc: {:4f}'.format(best_acc['flow']))

    ## for joint model
    i3d_rgb.load_state_dict(best_model_wts['joint']['rgb'])
    i3d_flow.load_state_dict(best_model_wts['joint']['flow'])
    torch.save(i3d_rgb.state_dict(), os.path.join(logpath, '{}_{}_bestiters_joint_{}.pth'.format('handhygiene', 'i3d_rgb', best_iters['joint'])))
    torch.save(i3d_flow.state_dict(), os.path.join(logpath, '{}_{}_bestiters_joint_{}.pth'.format('handhygiene', 'i3d_flow', best_iters['joint'])))

    ## for rgb/flow model
    i3d_rgb.load_state_dict(best_model_wts['rgb'])
    torch.save(i3d_rgb.state_dict(), 
               os.path.join(logpath, '{}_{}_bestiters_{}.pth'.format('handhygiene', 'i3d_rgb', best_iters['rgb'])))

    i3d_flow.load_state_dict(best_model_wts['flow'])
    torch.save(i3d_flow.state_dict(), 
               os.path.join(logpath, '{}_{}_bestiters_{}.pth'.format('handhygiene', 'i3d_flow', best_iters['flow'])))

    writer.close()
    manager.shutdown()
    torch.cuda.empty_cache()
    return 