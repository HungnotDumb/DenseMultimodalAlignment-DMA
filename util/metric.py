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
        conf