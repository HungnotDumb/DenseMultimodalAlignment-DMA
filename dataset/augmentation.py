
import random

import logging
import numpy as np
import scipy
import scipy.ndimage
import scipy.interpolate
import torch


# A sparse tensor consists of coordinates and associated features.
# You must apply augmentation to both.
# In 2D, flip, shear, scale, and rotation of images are coordinate transformation
# color jitter, hue, etc., are feature transformations
##############################
# Feature transformations
##############################
class ChromaticTranslation(object):
    '''Add random color to the image, input must be an array in [0,255] or a PIL image'''

    def __init__(self, trans_range_ratio=1e-1):
        '''
        trans_range_ratio: ratio of translation i.e. 255 * 2 * ratio * rand(-0.5, 0.5)
        '''
        self.trans_range_ratio = trans_range_ratio

    def __call__(self, coords, feats, labels):
        if random.random() < 0.95:
            tr = (np.random.rand(1, 3) - 0.5) * 255 * 2 * self.trans_range_ratio
            feats[:, :3] = np.clip(tr + feats[:, :3], 0, 255)
        return coords, feats, labels


class ChromaticAutoContrast(object):

    def __init__(self, randomize_blend_factor=True, blend_factor=0.5):
        self.randomize_blend_factor = randomize_blend_factor
        self.blend_factor = blend_factor

    def __call__(self, coords, feats, labels):