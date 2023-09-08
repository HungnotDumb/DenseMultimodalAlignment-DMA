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


    # short hand for processing 2D features
    scene = join(args.data_root_2d, scene_id)
    img_dirs = sorted(glob(join(scene, 'color/*')), key=lambda x: int(os.path.basename(x)[:-4]))
    num_img = len(img_dirs)
    device = torch.device('cpu')
    
    # load the text descriptions and extract the features
    # labelset = list(np.load(join('/home/liruihuang/openscene/data/tags_output_new', scene_id + '.npy'), allow_pickle=True).item()['true_list'])
    
    # print(labelset)
    # labelset.append('other')
    # args.text_features = extract_clip_feature(labelset, model_name="ViT-L/14@336px").cpu()
    # text_path = scene.replace('scannet_2d', 'scannet_clip_openseg_gpt_tag2text_labels/train') + '.pt'
    # torch.save(args.text_features, text_path)

    if torch.isnan(args.text_features).any():
        import pdb; pdb.set_trace()
  
    # extract image features and keep them in the memory
    # default: False (extract image on the fly)
    if keep_features_in_memory and openseg_model is not None:
        img_features = []
        for img_dir in tqdm(img_dirs):
            img_features.append(extract_openseg_img_feature(img_dir, openseg_model, text_emb, img_size=[240, 320]))

    n_points_cur = n_points
    counter = torch.zeros((n_points_cur, 1), device=device)
    sum_features = torch.zeros((n_points_cur, feat_dim), device=device)
    n_classes = len(args.text_features)
    pred_cls_num = torch.zeros((n_points_cur, n_classes), device=device)

    ################ Feature Fusion ###################
    vis_id = torch.zeros((n_points_cur, num_img), dtype=int, device=device)
    for img_id, img_dir in enumerate(tqdm(img_dirs)):
        # load pose
        posepath = img_dir.replace('color', 'pose').replace('.jpg', '.txt')
        pose = np.loadtxt(posepath)

        # load depth and convert to meter
        depth = imageio.v2.imread(img_dir.replace('color', 'depth').replace('jpg', 'png')) / depth_scale

        # calculate the 3d-2d mapping based on the depth
        mapping = np.ones([n_points, 4], dtype=int)
        mapping[:, 1:4] = point2img_mapper.compute_mapping(pose, locs_in, depth)
        if mapping[:, 3].sum() == 0: # no points corresponds to this image, skip
            continue

        mapping = torch.from_numpy(mapping).to(device)
        mask = mapping[:, 3]
        vis_id[:, img_id] = mask

        semantic_mask = torch.from_numpy(np.load(img_dir.replace('scannet_2d', 'scannet_2d_fcclip').replace('jpg', 'npy')))

        sam_mask = np.array(Image.open(img_dir.replace('scannet_2d', 'scannet_2d_paint').replace('color/', '').replace('jpg', 'png')), dtype=np.int16)
        sam_mask = num_to_natural(F.interpolate(torch.from_numpy(sam_mask).unsqueeze(0).unsqueeze(1).float(), scale_factor=0.5).int().squeeze().numpy())
        
        semantic_sam_mask = semantic_mask.clone()
        num_sam_mask = len(np.unique(sam_mask))
        for i in range(num_sam_mask):
            cls_tmp, cls_num = np.unique(semantic_mask[sam_mask==i], return_counts=True)
            # print(f'cls_num of the {i}-th mask is {cls_num}')
            if len(cls_num)>0:
                semantic_sam_mask[sam_mask==i] = cls_tmp[np.argmax(cls_num)]            
        labe