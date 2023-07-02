import os
import torch
import imageio
import argparse
from os.path import join, exists
import numpy as np
from glob import glob
from tqdm import tqdm, trange
import tensorflow as tf2
import tensorflow.compat.v1 as tf
from fusion_util import extract_openseg_img_feature, PointCloudToImageMapper, save_fused_feature, get_matterport_camera_data
import torch.nn.functional as F
import open3d as o3d
os.environ["CUDA_VISIBLE_DEVICES"] = "8"

def get_args():
    '''Command line arguments.'''

    parser = argparse.ArgumentParser(
        description='Multi-view feature fusion of OpenSeg on Matterport3D.')
    parser.add_argument('--data_dir', type=str, default='/home/liruihuang/openscene/data_matterport', help='Where is the base logging directory')
    parser.add_argument('--output_dir', type=str, default='./output', help='Where is the base logging directory')
    parser.add_argument('--split', type=str, default='train', help='split: "train"| "val" | "test" ')
    parser.add_argument('--openseg_model', type=str, default='', help='Where is the exported OpenSeg model')
    parser.add_argument('--process_id_range', nargs='+', default=None, help='the id range to process')
    parser.add_argument('--img_feat_dir', type=str, default='', help='the id range to process')

    # Hyper parameters
    parser.add_argument('--hparams', default=[], nargs="+")
    args = parser.parse_args()
    return ar