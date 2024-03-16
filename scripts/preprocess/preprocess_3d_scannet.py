import glob, os
import multiprocessing as mp
import numpy as np
import plyfile
import torch
from helper_ply import *
import pdb
# Map relevant classes to {0,1,...,19}, and ignored classes to 255
remapper = np.ones(150) * (255)
for i, x in enumerate([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 24, 28, 33, 34, 36, 39]):
    remapper[x] = i


def process_one_scene(fn):
    '''process one sc