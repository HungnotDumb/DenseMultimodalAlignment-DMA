import os
import math
import multiprocessing as mp
import numpy as np
import imageio
import cv2

def process_txt(filename):
    with open(filename) as file:
        lines = file.readlines()
        lines = [line.rstrip() for line in lines]
    return lines

def adjust_intrinsic(intrinsic, intrinsic_image_dim, image_dim):
    if intrinsic_ima