
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


def get_args():
    '''Command line arguments.'''

    parser = argparse.ArgumentParser(
        description='Multi-view feature fusion of OpenSeg on Matterport3D.')
    parser.add_argument('--data_dir', type=str, help='Where is the base logging directory')
    parser.add_argument('--output_dir', type=str, help='Where is the base logging directory')
    parser.add_argument('--split', type=str, default='test', help='split: "train"| "val" | "test" ')
    parser.add_argument('--openseg_model', type=str, default='', help='Where is the exported OpenSeg model')
    parser.add_argument('--process_id_range', nargs='+', default=None, help='the id range to process')
    parser.add_argument('--img_feat_dir', type=str, default='', help='the id range to process')

    # Hyper parameters
    parser.add_argument('--hparams', default=[], nargs="+")
    args = parser.parse_args()
    return args


def process_one_scene(data_path, out_dir, args):
    '''Process one scene.'''

    # short hand
    num_rand_file_per_scene = args.num_rand_file_per_scene
    feat_dim = args.feat_dim
    point2img_mapper = args.point2img_mapper
    depth_scale = args.depth_scale
    openseg_model = args.openseg_model
    text_emb = args.text_emb
    keep_features_in_memory = args.keep_features_in_memory

    # load 3D data (point cloud, color and the corresponding labels)
    locs_in = torch.load(data_path)[0]
    n_points = locs_in.shape[0]

    # obtain all camera views related information (specificially for Matterport)
    intrinsics, poses, img_dirs, scene_id, num_img = \
            get_matterport_camera_data(data_path, locs_in, args)
    if num_img == 0:
        print('no views inside {}'.format(scene_id))
        return 1

    n_interval = num_rand_file_per_scene
    n_finished = 0
    for n in range(n_interval):

        if exists(join(out_dir, scene_id +'_%d.pt'%(n))):
            n_finished += 1
            print(scene_id +'_%d.pt'%(n) + ' already done!')
            continue
    if n_finished == n_interval:
        return 1
