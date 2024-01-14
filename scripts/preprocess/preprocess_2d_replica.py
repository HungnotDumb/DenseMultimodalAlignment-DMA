
import glob, os
import multiprocessing as mp
import numpy as np
import imageio
import cv2
from tqdm import tqdm
from preprocess_util import make_intrinsic, adjust_intrinsic

def process_one_scene(fn):
    '''process one scene.'''

    # process RGB images
    img_name = fn.split('/')[-1]
    img_id = int(int(img_name.split('frame')[-1].split('.')[0])/sample_freq)
    img = imageio.v3.imread(fn)
    img = cv2.resize(img, img_dim, interpolation=cv2.INTER_LINEAR)
    imageio.imwrite(os.path.join(out_dir_color, str(img_id)+'.jpg'), img)

    # process depth images
    depth_name = img_name.replace('.jpg', '.png').replace('frame', 'depth')
    fn_depth = os.path.join(fn.split('frame')[0], depth_name)
    depth = imageio.v3.imread(fn_depth).astype(np.uint16)
    depth = cv2.resize(depth, img_dim, interpolation=cv2.INTER_LINEAR)
    imageio.imwrite(os.path.join(out_dir_depth, str(img_id)+'.png'), depth)

    #process poses
    np.savetxt(os.path.join(out_dir_pose, str(img_id)+'.txt'), pose_list[img_id])
    
def process_txt(filename):
    with open(filename) as file:
        lines = file.readlines()
        lines = [line.rstrip() for line in lines]
    return lines

#! YOU NEED TO MODIFY THE FOLLOWING
#####################################
out_dir = '../../data/replica_processed/replica_2d/'
in_path = '../../data/Replica/' # downloaded original replica data
sample_freq = 10
#####################################

os.makedirs(out_dir, exist_ok=True)

####### Meta Data #######