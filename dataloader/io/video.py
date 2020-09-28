import os
import re
import gc
import json
import torch
import numpy as np
import pandas as pd
from glob import glob
from PIL import Image
#from ..poseroi import bb_intersection_over_union



# TODO: all 2d bbox to 3d tubelet

def get_frames(dirname):
    from torchvision.datasets.folder import is_image_file
    return sorted([os.path.join(dirname, file) 
                   for file in os.listdir(dirname) 
                   if is_image_file(file)])

def read_video(dirname, start_pts=0, end_pts=None, has_bbox=False, 
               downsample=None, annotation=None, detection=None):
    frames = get_frames(dirname)
    video = []
    
    w = 0
    h = 0
    for i, frame in enumerate(frames):
        with open(frame, 'rb') as f:
            img = Image.open(f)
            img = img.convert('RGB')
            
            w, h = np.asarray(img).shape[:2]
            if downsample:
                ratio = min(w/downsample, h/downsample)
                w /= ratio
                h /= ratio
                img.thumbnail((h, w), Image.ANTIALIAS)
                #img.save(frame, "PNG")
            img = np.asarray(img)
            video.append(img)

    if end_pts is None:
        end_pts = len(video)

    if end_pts < start_pts:
        raise ValueError("end_pts should be larger than start_pts, got "
                         "start_pts={} and end_pts={}".format(start_pts, end_pts))
    
    video = np.asarray(video)
    video = torch.tensor(video)
    audio = torch.tensor([]) #tmp
    info = {'clip': dirname,
            'width': w,
            'height': h,
            'video_fps': 15.0,
           'keypoints': None,
           'annotations': []}
    
    if has_bbox:
        """ info_ann : a list of _get_bbox_info """
        info_ann = _set_info_annotation(dirname, annotation)
            
        if detection is None:
            info['annotations'] = info_ann
        else:
            video_path = os.path.basename(dirname)
            tubelet = [tubelet[1] for tubelet in detection if tubelet[0] == video_path][0]
            info_ann = tubelet_to_bbox(tubelet) 
            info['annotations'] = info_ann
            
    #info['annotations'] = _get_completed_annotations(info['annotations'])
    sample = (video, audio, info)
    return read_video_as_clip(sample, start_pts, end_pts, has_bbox)

#------------------------------------------------------------------------------#

# TODO: info는 tubelet으로 구성하기 
def _set_info_annotation(dirname, annotation):
    info_ann = []
    frames = get_frames(dirname)
    for frame in frames:
        image_path = os.path.splitext(os.path.basename(frame))[0]
        bboxes = _get_bbox_info(image_path, annotation)
        info_ann.append(bboxes)
    return info_ann
    
def target_dataframe(path):
    import pandas as pd
    df = pd.read_csv(path)
    df = df[df['person_id'].notnull()] # target 있는 imgpath만 선별
    return df

#------------------------------------------------------------------------------#

def read_video_as_clip(sample, start_pts, end_pts, has_bbox, target=None):
    video, audio, info = sample
    video = video[start_pts:end_pts+1]
    
    if end_pts == len(video):
        return (video, audio, info)
    
    if has_bbox:
        ##TODO: slicing keypoints
        #info['keypoints'] = info['keypoints'][start_pts:end_pts+1]
        info_tmp = info.copy()
        info_tmp['annotations'] = info_tmp['annotations'][start_pts:end_pts+1]
        
        if target is not None:
            info_tmp['annotations'] = _get_target_annotations(info_tmp['annotations'], target)
        
        #info_tmp['annotations'] = _get_completed_annotations(info_tmp['annotations'])
        return (video, audio, info_tmp)
    
    return (video, audio, info)


def read_video_timestamps(dirname):
    """ tmp function """
    frames = get_frames(dirname)
    return (list(range(len(frames)*1)), 15.0)


def _get_bbox_info(image_path, annotation, target_actions=[]):
    """ Given a image path and target actions, 
        Return all bboxes in a video clip """
    df = annotation
    
    if 'flow' in image_path:
        image_path = image_path.replace('_flow', '')
    
    # get all person annotations in a image path
    rows = df[df['image_path']==image_path]
    if len(rows) == 0:
        print("{} has no bounding box".format(image_path))
        return {}
    
    box_dict = {}
    for row in rows.values:
        _, _, _, label, person_id, x1, y1, w, h = row
        
        # exclude non-target action
        if len(target_actions) > 0:
            if label not in target_actions:
                continue
        box_dict[person_id] = [[x1, y1, w, h], label]
    return box_dict

def tubelet_to_bbox(tubelet):
    """ Convert tubelet to bbox dict similar to _get_bbox_info """
    info_ann=[]
    len_tube = len(tubelet[list(tubelet.keys())[0]][0])
    for i in range(len_tube):
        box_dict = {}
        for pid, tube in tubelet.items():
            bbox, label = tube
            x1, y1, w, h = bbox[i]
            box_dict[pid] = [[x1, y1, w, h], label]
        info_ann.append(box_dict)
    return info_ann

def _get_action_name(fname, person_id, annotation):
    """ return action label from data frame """
    
    df = annotation
    row = df[(df['clip_name'] == fname) & (df['person_id'] == person_id)]
    assert len(row) <= 1, "There is(are) {} labels on person {} in {}".format(len(row), person_id, fname)
    
    if len(row) == 1:
        # TODO:
        if row['action_name'].values[0] == 'touching_patient':
            return 'other_action'
        else:
            return row['action_name'].values[0]
    else: # no label
        return 'other_action'
    
    
def _get_bounding_box(fname, jsonfile, annotation, bbox_type='labelme'):
    ''' feeding action label here '''
    
    import json
    if bbox_type == 'labelme':
        bboxes = {}
        
        with open(jsonfile, 'r') as f:
            item = json.load(f)

            assert os.path.splitext(item['imagePath'])[0] == os.path.basename(os.path.splitext(jsonfile)[0])

            for person in item['shapes']:
                person_id = person['group_id']
                bbox = list(np.array(person['points']).reshape((4)))

                if person_id not in bboxes:
                    label = _get_action_name(fname, person_id, annotation)
                    bboxes[person_id] = [bbox, label]
    #TODO: else
    return bboxes


def _get_person_ids(annotations):
    person_ids = []
    for ann in annotations:
        for person_id in ann.keys():
            if person_id not in person_ids:
                person_ids.append(person_id)
    return sorted(person_ids)


def _get_action_tubelet(info):
    person_ids = info[0].keys()
    labels = {k: v[1] for k,v in info[0].items()}
    persons = {pid: [] for pid in person_ids}
    
    for iidx in info:
        
        for pid in person_ids:
            bbox, _ = iidx[pid]
            persons[pid].append(bbox)
    
    return {pid: [persons[pid], labels[pid]] for pid in person_ids}


def _get_action_tubelet_w_detection(video_path, info_ann, detection):
    
    video_path = os.path.basename(video_path)
    df_tmp = detection
    gt_tubelets = _get_action_tubelet(info_ann)
    row = df_tmp[df_tmp['video_path'] == video_path]
    
    # nan 너무 많으면 버리기.
    row = drop_non_target(row)
    dt_tubelets = get_all_tubelets(row) #allocate_iou_coco(row)
    dt_tubelets_dict = {}
    
    for tid, tubelet in enumerate(np.transpose(dt_tubelets, (1,0,2))):
        empty_count = (np.count_nonzero(tubelet == np.zeros(4,))/4)
        if  empty_count* 2 > len(tubelet):
            continue
            
        tubelet = calc_roi(tubelet, smoothing='linear')
        ious_3d = [bb_intersection_over_union_3d(gt_tubelets[pid][0], tubelet)
                   for pid in gt_tubelets.keys()]
        ious_3d_mean = [np.mean(bb_intersection_over_union_3d(gt_tubelets[pid][0], tubelet))
                        for pid in gt_tubelets.keys()]
        
        if len(ious_3d_mean) < 1 or max(ious_3d_mean) == 0:
            continue
        else:
            pid_new = list(gt_tubelets.keys())[np.argmax(ious_3d_mean)]
            
            dt_tubelets_dict[tid] = [[], '']
            dt_tubelets_dict[tid][0] = tubelet.tolist()
            dt_tubelets_dict[tid][1] = gt_tubelets[pid_new][1]
    return dt_tubelets_dict
    
#---------------------------------------------------------------------------------#
# With 2D Person Detector

def drop_non_target(row):
    '''Drop non target from ground truth'''
    
    num_frames = len(row['image_path'].drop_duplicates())
    pids = row['person_id'].drop_duplicates().values
    pids = [pid for pid in pids 
            if (row['person_id']==pid).value_counts()[1] > int(num_frames/2)]
    return row[row['person_id'].isin(pids)]

# TODO: switch to another temporal smoothing method

def calc_roi(rois, smoothing=None):

    buffer = rois  
    if smoothing == 'sma':
        buffer = moving_average(buffer, 4)
    elif smoothing == 'linear':
        buffer = np.array([[np.nan]*4 if np.max(roi) == 0. else roi for roi in buffer])
        buffer = interpolate_nan(buffer)
    return buffer


def moving_average(rois, period):
    buffer = np.zeros((len(rois), 4), dtype=int)
    for n, signal in enumerate(np.array(rois).T):
        for i in range(len(signal)):
            if i < period:
                buffer[i][n]=signal[i]
            else:
                buffer[i][n]=int(np.round(signal[i-period:i].mean()))
    return buffer


def interpolate_nan(rois):
    buffer = np.zeros_like(np.array(rois))
    for axis, y in enumerate(np.array(rois).T):
        nans, x= nan_helper(y)
        y[nans]= np.interp(x(nans), x(~nans), y[~nans])
        buffer[:,axis] = y
    return buffer

def nan_helper(y):
    return np.isnan(y), lambda z: z.nonzero()[0]


def bb_intersection_over_union(boxA, boxB):
    x11, y11, w1, h1 = boxA
    x21, y21, w2, h2 = boxB
    x12, y12 = x11+w1, y11+h1
    x22, y22 = x21+w2, y21+h2
    xA = np.maximum(x11, np.transpose(x21))
    yA = np.maximum(y11, np.transpose(y21))
    xB = np.minimum(x12, np.transpose(x22))
    yB = np.minimum(y12, np.transpose(y22))
    interArea = np.maximum((xB - xA + 1), 0) * np.maximum((yB - yA + 1), 0)
    boxAArea = (x12 - x11 + 1) * (y12 - y11 + 1)
    boxBArea = (x22 - x21 + 1) * (y22 - y21 + 1)
    iou = interArea / (boxAArea + np.transpose(boxBArea) - interArea)
    
    #print(boxAArea, boxBArea, interArea, iou)
    # return the intersection over union value
    return iou


def bb_intersection_over_union_3d(boxes0, boxes1, thres=0.2):
    assert len(boxes0) == len(boxes1), print(len(boxes0), len(boxes1))
    ious = []
    for fid in range(len(boxes0)):
        iou = bb_intersection_over_union(boxes0[fid], boxes1[fid])
        ious.append(iou)
    if np.mean(ious) < thres:
        ious = [0] * len(boxes0)
    return ious


def get_all_tubelets(row, thres = 0.2):
    
    image_ids          = row['image_id'].drop_duplicates().values
    num_max_detections = np.max(row['image_id'].value_counts())
    num_frames         = len(image_ids)
    tubelet_np         = np.zeros((num_frames, num_max_detections, 4))

    for i, iid in enumerate(image_ids):

        def get_last_bbox(tubelet_np):
            """ Get the last non-empty bbox of each person detections """
            last_bboxes = np.zeros_like(tubelet_np[0])
            for pid, person in enumerate(np.transpose(tubelet_np, (1, 0, 2))):
                sh = person[0].shape
                bbox = [p for p in person[::] if not np.all(p == np.zeros(sh))]
                last_bboxes[pid] = np.array(bbox[0]) if len(bbox) > 0 else np.zeros(sh)
            return last_bboxes

        def get_indices(ious_per_frame, thres):
            """ Get indices with a maxiou """
            a = np.array(ious_per_frame)
            mask = np.logical_and(a == a.max(axis=0) , (a > thres))
            a[mask]=1
            a[~mask]=0
            return np.array([np.argmax(row) if np.max(row) > 0. else np.nan for row in a])

        detections = row[row['image_id'] == iid]
        columns    = ['x1','y1','w','h'] 
        # init
        if i == 0: 
            for pid, person in enumerate(detections[columns].values):
                x, y, w, h = person
                tubelet_np[i][pid] = np.array([x,y,w,h])
        else:
            ious_per_frame = np.zeros((num_max_detections, num_max_detections))
            last_boxes = get_last_bbox(tubelet_np)

            ## to calc ious
            for pid, person in enumerate(detections[columns].values):
                x, y, w, h = person
                ious = [bb_intersection_over_union(last_box, [x, y, w, h]) for last_box in last_boxes]
                ious_per_frame[pid] = np.array(ious)
            indices = get_indices(ious_per_frame, thres)

            ## allocate person
            for pid, person in enumerate(detections[columns].values):
                pid_new = indices[pid]
                x, y, w, h = person

                if pid_new is not np.nan:
                    try:
                        tubelet_np[i][int(indices[pid])] = np.array([x,y,w,h])
                    except:
                        tubelet_np[i][int(pid)] = np.array([x,y,w,h])
    return tubelet_np



#---------------------------------------------------------------------------------## Visualization

def visualize_bbox(video, tubelets, label='person', color='black'):
    import matplotlib.patches as patches
    import matplotlib.pyplot as plt

    for fi, img in enumerate(video):
        fig, ax = plt.subplots(1, figsize=(10,7))
        ax.imshow(img)
        ax.axis('off')

        for k, v in tubelets.items():
            x, y, w, h = v[0][fi]
            rect = patches.Rectangle((x,y), w, h, linewidth=3, edgecolor=color, facecolor='none')
            ax.annotate('{}{}'.format(label,k), (x+40, y-15), color='white', 
                    fontsize=10, ha='center', va='top',
                    bbox=dict(boxstyle="square", fc="black"))
            ax.add_patch(rect)
    return

def visualize_bbox_result(video, tubelets_gt, tubelets_det, annot=True):
    import matplotlib.patches as patches
    import matplotlib.pyplot as plt
    
    for fi, img in enumerate(video):
        fig, ax = plt.subplots(1, figsize=(10,7))
        ax.imshow(img)
        ax.axis('off')

        for k, v in tubelets_gt.items():
            x, y, w, h = v[0][fi]
            rect = patches.Rectangle((x,y), w, h, linewidth=3, edgecolor='red', facecolor='none')
            if annot:
                ax.annotate('ground_truth{}'.format(k), (x+40, y-15), color='white', 
                        fontsize=10, ha='center', va='top',
                        bbox=dict(boxstyle="square", fc="black"))
                ax.annotate('GT:{}'.format(v[1]), (x+40, y+h-15), color='black', 
                        fontsize=10, ha='center', va='top',
                        bbox=dict(boxstyle="square", fc="white"))
            ax.add_patch(rect)
            
        for k, v in tubelets_det.items():
            x, y, w, h = v[0][fi]
            rect = patches.Rectangle((x,y), w, h, linewidth=2, edgecolor='cyan', facecolor='none')
            if annot:
                ax.annotate('detection{}'.format(k), (x+40, y+15), color='white', 
                        fontsize=10, ha='center', va='bottom',
                        bbox=dict(boxstyle="square", fc="black"))
                ax.annotate('Result:{}'.format(v[1]), (x+40, y+h+15), color='black', 
                        fontsize=10, ha='center', va='bottom',
                        bbox=dict(boxstyle="square", fc="white"))
            ax.add_patch(rect)
    return

#-----------------------------------------------------------------------#

def _get_target_annotations(annotations, target):
    """Grab only targeted annotations throughout the video clip"""
    
    info_new = []
    for ann in annotations:
        ann_new = {}
        for person_id in ann.keys():
            if person_id == target:
                ann_new[person_id] = ann[person_id]
        info_new.append(ann_new)
    return info_new
