
'''Dataloader for fused point features.'''

import copy
from glob import glob
from os.path import join
import torch
import numpy as np
import SharedArray as SA

from dataset.point_loader import Point3DLoader

class FusedFeatureLoader(Point3DLoader):
    '''Dataloader for fused point features.'''

    def __init__(self,
                 datapath_prefix,
                 datapath_prefix_feat,
                 voxel_size=0.05,
                 split='train', aug=False, memcache_init=False,
                 identifier=7791, loop=1, eval_all=False,
                 input_color = False,
                 ):
        super().__init__(datapath_prefix=datapath_prefix, voxel_size=voxel_size,
                                           split=split, aug=aug, memcache_init=memcache_init,
                                           identifier=identifier, loop=loop,
                                           eval_all=eval_all, input_color=input_color)
        self.aug = aug
        self.input_color = input_color # decide whether we use point color values as input

        # prepare for 3D features
        self.datapath_feat = datapath_prefix_feat

        # Precompute the occurances for each scene
        # for training sets, ScanNet and Matterport has 5 each, nuscene 1
        # for evaluation/test sets, all has just one
        if 'nuscenes' in self.dataset_name: # only one file for each scene
            self.list_occur = None
        else:
            self.list_occur = []
            for data_path in self.data_paths:
                if 'scannet' in self.dataset_name and 'scannet_200' not in self.dataset_name:
                    scene_name = data_path[:-15].split('/')[-1]
                else:
                    scene_name = data_path[:-4].split('/')[-1]
                file_dirs = glob(join(self.datapath_feat, scene_name + '_*.pt'))
                self.list_occur.append(len(file_dirs))
            # some scenes in matterport have no features at all
            ind = np.where(np.array(self.list_occur) != 0)[0]

            if np.any(np.array(self.list_occur)==0):
                data_paths, list_occur = [], []
                for i in ind:
                    data_paths.append(self.data_paths[i])
                    list_occur.append(self.list_occur[i])
                self.data_paths = data_paths
                self.list_occur = list_occur

        if len(self.data_paths) == 0:
            raise Exception('0 file is loaded in the feature loader.')

    def __getitem__(self, index_long):
        index = index_long % len(self.data_paths)
        if self.use_shm:
            locs_in = SA.attach("shm://%s_%s_%06d_locs_%08d" % (
                self.dataset_name, self.split, self.identifier, index)).copy()
            feats_in = SA.attach("shm://%s_%s_%06d_feats_%08d" % (
                self.dataset_name, self.split, self.identifier, index)).copy()
            labels_in = SA.attach("shm://%s_%s_%06d_labels_%08d" % (
                self.dataset_name, self.split, self.identifier, index)).copy()
        else:
            if 'scannet_200' in self.dataset_name:
                data = torch.load(self.data_paths[index])
                locs_in = data['coord']
                feats_in = data['color']
                labels_in = data['semantic_gt200']   
                paint_label_path = self.data_paths[index].replace('scannet_200', 'scannet_200_sam_paint').replace('_vh_clean_2.pth','.pth')
            else:
                locs_in, feats_in, labels_in = torch.load(self.data_paths[index])
                paint_label_path = self.data_paths[index].replace('nuscenes_3d/val', 'nuscenes_2d/nuscenes_3d_fcclip_paint')
            
            # if self.split == 'train':
            paint_labels_in = torch.from_numpy(torch.load(paint_label_path))

            labels_in[labels_in == -100] = 255
            labels_in = labels_in.astype(np.uint8)
            if np.isscalar(feats_in) and feats_in == 0:
                # no color in the input point cloud, e.g nuscenes lidar
                feats_in = np.zeros_like(locs_in)
            else: