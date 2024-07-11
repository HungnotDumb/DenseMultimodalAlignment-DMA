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

if not os.path.exists(opt.output_pat