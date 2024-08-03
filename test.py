import os
import torch
import numpy as np
from glob import glob
from os.path import join, exists
from tqdm import tqdm, trange
from util.util import export_pointcloud
from plyfile import PlyData, PlyElement

# data_paths = './data_matterport/matterport_multiview_openseg_test/pa4otMbVnkk_region8_0.pt'
# scene_name = 'YVUC4YcDtcY'
# file_dirs = sorted(glob(join('./data_matterport/matterport_3d/test', scene_name + '_*.pth')))
# pcl_list=[]
# color_list = []

# for file in file_dirs:
#     pc = torch.load(file)
#     pcl_list.append(pc[0])
#     color_list.append(pc[1])

# pcl = np.concatenate(pcl_list, axis=0)
# color = np.concatenate(color_list, axis=0)
# # import pdb; pdb.set_trace()

# mask = pcl[:,2]<4.0

# pcl = pcl[mask]
# color = color[mask]

# color = (color+1)/2
# save_path = join('./data_matterport/visualization', scene_name)
# if not os.path.exists(save_path):
#     os.makedirs(save_path)
pc = torch.load('./data/scannet_3d/train/scene0464_00_vh_clean_2.pth')
pcl=pc[0]
color = (pc[1]+1)/2

export_pointcloud('0464.ply', pcl, colors=color)
import pdb; pdb.set_trace()
# file_dirs_1 = [file.replace('.pth', '_distill.npy').replace('data_matterport/matterport_3d/test','save_matter_openscene') for file in file_dirs]
# # file_dirs = sorted(glob(join('./save_matter_openscene', scene_name + '_*_distill.npy')))
# colo