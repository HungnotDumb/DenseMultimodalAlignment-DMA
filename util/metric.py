'''IoU'''
import numpy as np
from dataset.label_constants import *

UNKNOWN_ID = 255
NO_FEATURE_ID = 256


def confusion_matrix(pred_ids, gt_ids, num_classes):
    '''calculate the confusion matrix.'''

    assert pred_ids.shape == gt_ids.shape, (pred_ids.shape, gt_ids.shape)
    idxs = gt_ids != UNKNOWN_ID
    if NO_FEATURE_ID in pred_ids: # some points have no feature assigned for prediction
        pred_ids[pred_ids==NO_FEATURE_ID] = num_classes
        confusion = np.bincount(
            pred_ids[idxs] * (num_classes+1) + gt_ids[idxs],
            minlength=(num_classes+1)**2).reshape((
            num_classes+1, num_classes+1)).astype(np.ulonglong)
        # return confusion[:num_classes, :num_classes]
        return confusion

    return np.bincount(
        pred_ids[idxs] * num_classes + gt_ids[idxs],
        minlength=num_classes**2).reshape((
        num_classes, num_classes)).astype(np.ulonglong)


def get_iou(label_id, confusion):
    '''calculate IoU.'''

    # true positives
    tp = np.longlong(confusion[label_id, label_id])
    # false positives
    fp = np.longlong(confusion[label_id, :].sum()) - tp
    # false negatives
    fn = np.longlong(confusion[:, label_id].sum()) - tp

    denom = (tp + fp + fn)
    if denom == 0:
        return float('nan')
    return float(tp) / denom, tp, denom

def get_iou_scannet200(label_id, confusion):
    '''calculate IoU.'''

    # true positives
    tp = np.longlong(confusion[label_id, label_id])
    # false positives
    fp = np.longlong(confusion[label_id, :].sum()) - tp
    # false negatives
    fn = np.longlong(confusion[:, label_id].sum()) - tp

    denom = (tp + fp + fn)
    if denom == 0:
        return float('nan')
    return float(tp) / denom, tp, denom, tp/(tp+fp+1e-5), tp/(tp+fn+1e-5)


def evaluate(pred_ids, gt_ids, stdout=False, dataset='scannet_3d'):
    if stdout:
        print('evaluating', gt_ids.size, 'points...')
    if 'head' in dataset:
        CLASS_LABELS = HEAD_CATS_SCANNET_200
    elif 'scannet_200' in dataset:
        CLASS_LABELS = SCANNET_LABELS_200
    elif 'scannet_3d' in dataset:
        CLASS_LABELS = SCANNET_LABELS_20
    elif 'matterport_3d_40' in dataset:
        CLASS_LABELS = MATTERPORT_LABELS_40
    elif 'matterport_3d_80' in dataset:
        CLASS_LABELS = MATTERPORT_LABELS_80
    elif 'matterport_3d_160' in dataset:
        CLASS_LABELS = MATTERPORT_LABELS_160
    elif 'matterport_3d' in dataset:
        CLASS_LABELS = MATTERPORT_LABELS_21
    elif 'nuscenes_3d' in dataset:
        CLASS_LABELS = NUSCENES_LABELS_16
    else:
        raise NotImplementedError

    N_CLASSES = len(CLASS_LABELS)
    confusion = confusion_matrix(pred_ids, gt_ids, N_CLASSES)
    class_ious = {}
    class_accs = {}
    mean_iou = 0
    mean_acc = 0

    count = 0
    for i in range(N_CLASSES):
        label_name = CLASS_LABELS[i]
        if (gt_ids==i).sum() == 0: # at least 1 point needs to be in the evaluatio