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
    return args


def process_one_scene(data_path, out_dir, args):
    '''Process one scene.'''

    # short hand
    num_rand_file_per_scene = args.num_rand_file_per_scene
    feat_dim = args.feat_dim
    point2img_mapper = args.point2img_mapper
    depth_scale = args.depth_scale


    # load 3D data (point cloud, color and the corresponding labels)
    locs_in = torch.load(data_path)[0]
    labels_in = torch.load(data_path)[2]
    n_points = locs_in.shape[0]

    # obtain all camera views related information (specificially for Matterport)
    intrinsics, poses, img_dirs, scene_id, num_img = \
            get_matterport_camera_data(data_path, locs_in, args)
    if num_img == 0:
        print('no views inside {}'.format(scene_id))
        return 1

    device = 'cuda'

    n_points_cur = n_points

    n_classes = 160
    pred_cls_num = torch.zeros((n_points_cur, n_classes), device=device)

    ################ Feature Fusion ###################
    vis_id = torch.zeros((n_points_cur, num_img), dtype=int, device=device)
    for img_id, img_dir in enumerate(tqdm(img_dirs)):
        # load pose
        pose = poses[img_id]

        # load per-image intrinsic
        intr = intrinsics[img_id]

        # load depth and convert to meter
        depth_dir = img_dir.replace('color', 'depth')
        _, img_type, yaw_id = img_dir.split('/')[-1].split('_')
        depth_dir = depth_dir[:-8] + 'd'+img_type[1] + '_' + yaw_id[0] + '.png'
        depth = imageio.v2.imread(depth_dir) / depth_scale

        # calculate the 3d-2d mapping based on the depth
        mapping = np.ones([n_points, 4], dtype=int)
        mapping[:, 1:4] = point2img_mapper.compute_mapping(pose, locs_in, depth, intr)
        if mapping[:, 3].sum() == 0: # no points corresponds to this image, skip
            continue
        mapping = torch.from_numpy(mapping).to(device)
        mask = mapping[:, 3]
        vis_id[:, img_id] = mask

        semantic_mask = torch.from_numpy(np.load(img_dir.replace('matterport_2d', 'matterport_semantic_label_fcclip').replace('color/','').replace('jpg', 'npy'))).to(device)

        label_one_hot = F.one_hot(semantic_mask.long(), n_classes)
        # img = imageio.v2.imread(img_dir)
        # visualize_2d(img, semantic_mask.numpy(), semantic_mask.shape, './semantic_mask.png')
        # visualize_partition_2d(semantic_sam_mask)
        # visualize_2d(img, semantic_sam_mask*(semantic_sam_m