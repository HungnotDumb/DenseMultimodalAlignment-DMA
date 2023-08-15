import os
os.environ["CUDA_VISIBLE_DEVICES"] = "9"
import sys
sys.path.append("/home/liruihuang/openscene")
import torch
import imageio
import argparse
import numpy as np
from glob import glob
from tqdm import tqdm, trange
import tensorflow as tf2
import tensorflow.compat.v1 as tf
from os.path import join, exists
from fusion_util import extract_openseg_img_feature, PointCloudToImageMapper, save_fused_feature
import torch.nn.functional as F
from PIL import Image
import copy
import open3d as o3d

from util.util import extract_clip_feature

def get_args():
    '''Command line arguments.'''

    parser = argparse.ArgumentParser(
        description='Multi-view feature fusion of OpenSeg on ScanNet.')
    parser.add_argument('--data_dir', type=str, help='Where is the base logging directory')
    parser.add_argument('--split', type=str, default='train', help='split: "train"| "val"')
    parser.ad