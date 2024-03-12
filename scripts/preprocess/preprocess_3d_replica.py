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
    torch.save((coords, colors, labels),
            os.path.join(out_dir,  scene_name + '.pth'))
    print(fn)


def process_txt(filename):
    with open(filename) as file:
        lines = file.readlines()
        lines = [line.rstrip() for line in lines]
    return lines

#! YOU NEED TO MODIFY THE FOLLOWING
scene_list = ['office0', 'office1', 'office2', 'office3',
              'office4', 'room0', 'room1', 'room2']
####################