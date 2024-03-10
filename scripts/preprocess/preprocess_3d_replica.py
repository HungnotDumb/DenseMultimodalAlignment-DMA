import os
import multiprocessing as mp
import numpy as np
import plyfile
import torch



def process_one_scene(fn):
    '''process one scene.'''

    scene_name = fn.split('/')[-1].split('_mesh')[0]
    a = plyfile.PlyData().read(fn)
    v = np.array([list(x) for x in a.elements[0]])
    coords = np.ascontiguousarray(v[:, :3])
    colors = np.ascontiguousarray(v[:, -3:]) / 127.5 - 1

    # no GT labels are provided, set all to 255
    labels = 255*np.ones((coords.shape[0], ), dtype=np.int32)
    torch.s