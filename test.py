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

# for file in fil