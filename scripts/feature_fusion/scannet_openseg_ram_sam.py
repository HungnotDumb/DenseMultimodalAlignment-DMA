
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "7"
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
 
    # for n in range(n_interval):

    #     if exists(join(out_dir, scene_id +'_%d.pt'%(n))):
    #         n_finished += 1
    #         print(scene_id +'_%d.pt'%(n) + ' already done!')
    #         continue
    # if n_finished == n_interval:
    #     return 1

    # short hand for processing 2D features
    scene = join(args.data_root_2d, scene_id)
    img_dirs = sorted(glob(join(scene, 'color/*')), key=lambda x: int(os.path.basename(x)[:-4]))
    num_img = len(img_dirs)
    device = torch.device('cpu')
    
    # load the text descriptions and extract the features
    labelset = list(np.load(join('/home/liruihuang/openscene/data/tags_output_new', scene_id + '.npy'), allow_pickle=True).item()['true_list'])
    
    # print(labelset)
    # labelset.append('other')
    # args.text_features = extract_clip_feature(labelset, model_name="ViT-L/14@336px").cpu()
    # text_path = scene.replace('scannet_2d', 'scannet_clip_openseg_gpt_tag2text_labels/train') + '.pt'
    # torch.save(args.text_features, text_path)