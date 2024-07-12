# Copyright (c) Facebook, Inc. and its affiliates.
# 
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import glob, os
import numpy as np
import cv2
import argparse

from plyfile import PlyData, PlyElement

# params
parser = argparse.ArgumentParser()
# data paths
parser.add_argument('--input_path', required=True, help='path to sens file to read')
parser.add_argument('--output_path', required=True, help='path to output folder')
parser.add_argument('--save_npz', action='store_true')
opt = parser.parse_args()
print(opt)

if not os.path.exists(opt.output_path):
    os.mkdir(opt.output_path)

# Load Depth Camera Intrinsic
depth_intrinsic = np.loadtxt(opt.input_path + '/intrinsic/intrinsic_depth.txt')
print('Depth intrinsic: ')
print(depth_intrinsic)

# Compute Camrea Distance (just for demo, so you can choose the camera distance in frame sampling)
poses = sorted(glob.glob(opt.input_path + '/pose/*.txt'), key=lambda a: int(os.path.basename(a).split('.')[0]))
depths = sorted(glob.glob(opt.input_path + '/depth/*.png'), key=lambda a: int(os.path.basename(a).split('.')[0]))
colors = sorted(glob.glob(opt.input_path + '/color/*.png'), key=lambda a: int(os.path.basename(a).split('.')[0]))

# # Get Aligned Point Clouds.
for ind, (pose, depth, color) in enumerate(zip(poses, depths, colors)):
    name = o