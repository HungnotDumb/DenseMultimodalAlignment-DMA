# Copyright (c) Facebook, Inc. and its affiliates.
# 
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import argparse
import glob, os, sys

from SensorData import SensorData

# params
parser = argparse.ArgumentParser()
# data paths
parser.add_argument('--target_dir', required=True, help='path to the target dir')

opt = parser.parse_args()
print(opt)

def