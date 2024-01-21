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
parser.add_argument('--label_type', default='label-filt', help='which labels (label or label-filt)')
parser.add_argument('--frame_skip', type=int, default=20, help='export every nth frame')
parser.add_argument('--label_map_file', default='',
                    help='path to scannetv2-labels.combined.tsv (required for label export only)')
parser.add_argument('--output_image_width', type=int, default=320, help='export image width')
parser.add_argument('--output_image_height', type=int, default=240, help='export image height')

parser.set_defaults(export_label_images=False)
opt = parser.parse_args()
if opt.export_label_images:
    assert opt.label_map_file != ''
print(opt)


def print_error(message):
    sys.stderr.write('ERROR: ' + str(message) + '\n')
    sys.exit(-1)

def map_label_image(image, label_mapping):
    mapped = np.copy(image)
    for k, v in label_mapping.items():
        mapped[image == k] = v
    return mapped.astype(np.uint8)

def represents_int(s):
    try:
        int(s)
        return True
    except ValueError:
        return False

def read_label_mapping(filename, label_from='raw_category', label_to='nyu40id'):
    assert os.path.isfile(filename)
    mapping = dict()
    with open(filename) as csvfile:
        reader = csv.Dic