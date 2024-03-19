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
    '''process one scene.'''

    fn2 = fn[:-3] + 'labels.ply'
    a = plyfile.PlyData().read(fn)
    v = np.array([list(x) for x in a.elements[0]])
    coords = np.ascontiguousarray(v[:, :3])
    
    # colors = np.ascontiguousarray(v[:, 3:6]) / 127.5 - 1
    colors = np.ascontiguousarray(v[:, 3:6]).astype(np.uint8)

    a = plyfile.PlyData().read(fn2)
    w = remapper[np.array(a.elements[0]['label'])]

    field_names = ['x', 'y', 'z', 'red', 'green', 'blue', 'values']

    write_ply(os.path.join(out_dir, fn[:-4].split('/')[-1] + '.ply'), [coords, colors, w], field_names)

    # torch.save((coords, colors, w),
    #         os.path.join(out_dir, fn[:-4].split('/')[-1] + '.pth'))
    print(fn, fn2)


def process_txt(filename):
    with open(filename) as file:
        lines = file.readlines()
      