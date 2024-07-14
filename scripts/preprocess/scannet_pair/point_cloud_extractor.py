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
    name = os.path.basename(pose).split('.')[0]

    if os.path.exists(opt.output_path + '/{}.npz'.format(name)):
        continue

    try:
        print('='*50, ': {}'.format(pose))
        depth_img = cv2.imread(depth, -1) # read 16bit grayscale image
        mask = (depth_img != 0)
        color_image = cv2.imread(color)
        color_image = cv2.resize(color_image, (640, 480))
        color_image = np.reshape(color_image[mask], [-1,3])
        colors = np.zeros_like(color_image)
        colors[:,0] = color_image[:,2]
        colors[:,1] = color_image[:,1]
        colors[:,2] = color_image[:,0]

        pose = np.loadtxt(poses[ind])
        print('Camera pose: ')
        print(pose)
        
        depth_shift = 1000.0
        x,y = np.meshgrid(np.linspace(0,depth_img.shape[1]-1,depth_img.shape[1]), np.linspace(0,depth_img.shape[0]-1,depth_img.shape[0]))
        uv_depth = np.zeros((depth_img.shape[0], depth_img.shape[1], 3))
        uv_depth[:,:,0] = x
        uv_depth[:,:,1] = y
        uv_depth[:,:,2] = depth_img/depth_shift
        uv_depth = np.reshape(uv_depth, [-1,3])
        uv_depth = uv_depth[np.where(uv_depth[:,2]!=0),:].squeeze()
        
        intrinsic_inv = np.linalg.inv(de