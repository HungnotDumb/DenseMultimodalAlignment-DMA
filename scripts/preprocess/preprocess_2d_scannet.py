# pre-process ScanNet 2D data
# code adapted from https://github.com/angeladai/3DMV/blob/master/prepare_data/prepare_2d_data.py
#
# note: depends on the sens file reader from ScanNet:
#       https://github.com/ScanNet/ScanNet/blob/master/SensReader/python/SensorData.py
# which is renamed to scannet_sensordata.py under this directory

# Example usage:
#    python prepare_2d_scannet.py --scannet_path /PATH_TO/scannet/scans \
#                             --output_path ../../data/scannet_2d \
#                             --export_label_images \
#                             --label_map_file /PATH_TO_TSV_FILE/scannetv2-labels.combined.tsv

import argparse
import os
import sys
import csv
import numpy as np
import skimage.transform as sktf
import imageio
from scannet_sensordata import SensorData
from preprocess_util import make_intrinsic, adjust_intrinsic

# params
parser = argparse.ArgumentParser()
parser.add_argument('--scannet_path', required=True, help='path to scannet data')
parser.add_argument('--output_path', required=True, help='where to output 2d data')
parser.add_argument('--export_label_images', dest='export_label_images', action='store_true')
parser.add_argument('--label_t