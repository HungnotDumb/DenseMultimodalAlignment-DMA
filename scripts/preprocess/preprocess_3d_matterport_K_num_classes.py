import glob, os
import multiprocessing as mp
import numpy as np
import plyfile
import torch
import pandas as pd



def process_one_scene(fn):
    '''process one scene.'''

    scene_name = fn.split('/')[-3]
    region_name = fn.split('/')[-1].split('.')[0]
    a = plyfile.PlyData().read(fn)
    v = np.array([list(x) for x in a.elements[0]])
    coords = np.ascontiguousarray(v[:, :3])
    colors = np.ascontiguousarray(v[:, -3:]) / 127.5 - 1

    category_id = a['face']['category_id']
    category_id[category_id==-1] = 0
    mapped_labels = mapping[category_id]

    triangles = a['face']['vertex_indices']
    vertex_labels = np.zeros((coords.shape[0], num_classes+1), dtype=np.int32)
    # calculate per-vertex labels
    for row_id in range(triangles.shape[0]):
        for i in range(3):
            vertex_labels[triangles[row_id][i],
                            mapped_labels[row_id]] += 1

    vertex_labels = np.argmax(vertex_labels, axis=1)
    vertex_labels[vertex_labels==0] = 256
    vertex_labels -= 1

    torch.save((coords, colors, vertex_labels),
            os.path.join(out_dir,  scene_name+'_' + region_name + '.pth'))
    print(fn)


def process_txt(filename):
    with open(filename) as file:
        lines = file.readlines()
        lines = [line.rstrip() for line in lines]
    return lines

#! YOU NEED TO MODIFY THE FOLLOWING
#####################################
split = 'test' # 'train' | 'val' | 'test'
num_classes = 160 # 40 | 80 | 160 # define the number of classes
out_dir = '../../data/matterport_3d_{}/{}'.format(num_classes, split)
matterport_path = '/PATH_TO/matterport/scans' # downloaded original matterport data
tsv_file = '../../dataset/matterport/category_mapping.tsv'
scene_list = process_txt('../../dataset/matterport/scenes_{}.txt'.format(split))
#####################################

os.makedirs(out_dir, exist_ok=True)
category_mapping 