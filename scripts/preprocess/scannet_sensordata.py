# code supports usages in Python3.
# Adapted from https://github.com/ScanNet/ScanNet/blob/master/SensReader/python/SensorData.py

import os, struct
import numpy as np
import zlib
import imageio
import cv2

COMPRESSION_TYPE_COLOR = {-1:'unknown', 0:'raw', 1:'png', 2:'jpeg'}
COMPRESSION_TYPE_DEPTH = {-1:'