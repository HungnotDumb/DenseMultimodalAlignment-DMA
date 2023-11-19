import os
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


def get_args():
    '''Command line arguments.'''

    parser = argparse.ArgumentParser(
        description='Multi-view feature fusion of OpenSeg on ScanNet.')
    parser.add_argument('--data_dir', type=str, help='Where is the base logging directory')
    parser.add_argument('--split', type=str, default='train', help='split: "train"| "val"')
    parser.add_argument('--output_dir', type=str, help='Where is the base logging directory')
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
    scene_id = data_path.split('/')[-1].split('_vh')[0]

    num_rand_file_per_scene = args.num_rand_file_per_scene
    feat_dim = args.feat_dim
    point2img_mapper = args.point2img_mapper
    depth_scale = args.depth_scale
    openseg_model = args.openseg_model
    text_emb = args.text_emb
    keep_features_in_memory = args.keep_features_in_memory

    # load 3D data (point cloud)
    locs_in = torch.load(data_path)[0]
    n_points = locs_in.shape[0]

    n_interval = num_rand_file_per_scene
    n_finished = 0
    for n in range(n_interval):

        if exists(join(out_dir, scene_id +'_%d.pt'%(n))):
            n_finished += 1
            print(scene_id +'_%d.pt'%(n) + ' already done!')
            continue
    if n_finished == n_interval:
        return 1

    # short hand for processing 2D features
    scene = join(args.data_root_2d, scene_id)
    img_dirs = sorted(glob(join(scene, 'color/*')), key=lambda x: int(os.path.basename(x)[:-4]))
    num_img = len(img_dirs)
    device = torch.device('cpu')

    # extract image features and keep them in the memory
    # default: False (extract image on the fly)
    if keep_features_in_memory and openseg_model is not None:
        img_features = []
        for img_dir in tqdm(img_dirs):
            img_features.append(extract_openseg_img_feature(img_dir, openseg_model, text_emb, img_size=[240, 320]))

    n_points_cur = n_points
    counter = torch.zeros((n_points_cur, 1), device=device)
    sum_features = torch.zeros((n_points_cur, feat_dim), device=device)

    ################ Feature Fusion ###################
    vis_id = torch.zeros((n_points_cur, num_im