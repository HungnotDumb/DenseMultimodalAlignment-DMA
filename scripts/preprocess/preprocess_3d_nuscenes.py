import os
import multiprocessing as mp
import numpy as np
import plyfile
import torch



NUSCENES_FULL_CLASSES = ( # 32 classes
    'noise',
    'animal',
    'human.pedestrian.adult',
    'human.pedestrian.child',
    'human.pedestrian.construction_worker',
    'human.pedestrian.personal_mobility',
    'human.pedestrian.police_officer',
    'human.pedestrian.stroller',
    'human.pedestrian.wheelchair',
    'movable_object.barrier',
    'movable_object.debris',
    'movable_object.pushable_pullable',
    'movable_object.trafficcone',
    'static_object.bicycle_rack',
    'vehicle.bicycle',
    'vehicle.bus.bendy',
    'vehicle.bus.rigid',
    'vehicle.car',
    'vehicle.construction',
    'vehicle.emergency.ambulance',
    'vehicle.emergency.police',
    'vehicle.motorcycle',
    'vehicle.trailer',
    'vehicle.truck',
    'flat.driveable_surface',
    'flat.other',
    'flat.sidewalk',
    'flat.terrain',
    'static.manmade',
    'static.other',
    'static.vegetation',
    'vehicle.ego',
    'unlabeled',
)

VALID_NUSCENES_CLASS_IDS = ()

NUSCENES_CLASS_REMAP = 256*np.ones(32) # map from 32 classes to 16 classes
NUSCENES_CLASS_REMAP[2] = 7 # person
NUSCENES_CLASS_REMAP[3] = 7
NUSCEN