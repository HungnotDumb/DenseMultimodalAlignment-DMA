#!/usr/bin/env python
# Downloads ScanNet public data release
# Run with ./download-scannet.py (or python download-scannet.py on Windows)
# -*- coding: utf-8 -*-
import argparse
import os
import urllib.request
# import urllib
import tempfile

BASE_URL = 'http://kaldir.vc.in.tum.de/scannet/'
TOS_URL = BASE_URL + 'ScanNet_TOS.pdf'
FILETYPES = ['.aggregation.json', '.sens', '.txt', '_vh_clean.ply', '_vh_clean_2.0.010000.segs.json', '_vh_clean_2.ply', '_vh_clean.segs.json', '_vh_clean.aggregation.json', '_vh_clean_2.labels.ply', '_2d-instance.zip', '_2d-instance-filt.zip', '_2d-label.zip', '_2d-label-filt.zip']
FILETYPES_TEST = ['.sens', '.txt', '_vh_clean.ply', '_vh_clean_2.ply']
PREPROCESSED_FRAMES_FILE = ['scannet_frames_25k.zip', '5.6GB']
TEST_FRAMES_FILE = ['scannet_frames_test.zip', '610MB']
LABEL_MAP_FILES = ['scannetv2-labels.combined.tsv', 'scannet-labels.combined.tsv']
DATA_EFFICIENT_FILES = ['limited-reconstruction-scenes.zip', 'limited-annotation-points.zip', 'limited-bboxes.zip', '1.7MB']
GRIT_FILES = ['ScanNet-GRIT.zip']
RELEASES = ['v2/scans', 'v1/scans']
RELEASES_TASKS = ['v2/tasks', 'v1/tasks']
RELEASES_NAMES = ['v2', 'v1']
RELEASE = RELEASES[0]
RELEASE_TASKS = RELEASES_TASKS[0]
RELEASE_NAME = RELEASES_NAMES[0]
LABEL_MAP_FILE = LABEL_MAP_FILES[0]
RELEASE_SIZE = '1.2TB'
V1_IDX = 1


def get_release_scans(release_file):
    scan_lines = urllib.request.urlopen(release_file)
    # scan_lines = urllib.urlopen(release_file)
    scans = []
    for scan_line in scan_lines:
        scan_id = scan_line.decode('utf8').rstrip('\n')
        scans.append(scan_id)
    return scans


def download_release(release_scans, out_dir, file_types, use_v1_sens):
    if len(release_scans) == 0:
        return
    print('Downloading ScanNet ' + RELEASE_NAME + ' release to ' + out_dir + '...')
    for scan_id in release_scans:
        scan_out_dir = os.path.join(out_dir, scan_id)
        download_scan(scan_id, scan_out_dir, file_types, use_v1_sens)
    print('Downloaded ScanNet ' + RELEASE_NAME + ' release.')


def download_file(url, out_file):
    out_dir = os.path.dirname(out_file)
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)
    if not os.path.isfile(out_file):
        print('\t' + url + ' > ' + out_file)
        fh, out_file_tmp = tempfile.mkstemp(dir=out_dir)
        f = os.fdopen(fh, 'w')
        f.close()
        urllib.request.urlretrieve(url, out_file_tmp)
        # urllib.urlretrieve(url, out_file_tmp)
        os.rename(out_file_tmp, out_file)
    else:
        print('WARNING: skipping download of existing file ' + out_file)

def download_scan(scan_id, out_dir, file_types, use_v1_sens):
    print('Downloading ScanNet ' + RELEASE_NAME + ' scan ' + scan_id + ' ...')
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)
    for ft in file_types:
        v1_sens = use_v1_sens and ft == '.sens'
        url = BASE_URL + RELEASE + '/' + scan_id + '/' + scan_id + ft if not v1_sens else BASE_URL + RELEASES[V1_IDX] + '/' + scan_id + '/' + scan_id + ft
        out_file = out_dir + '/' + scan_id + ft
        download_file(url, out_file)
    print('Downloaded scan ' + scan_id)


def download_task_data(out_dir):
    print('Downloading ScanNet v1 task data...')
    files = [
        LABEL_MAP_FILES[V1_IDX], 'obj_classification/data.zip',
        'obj_classification/trained_models.zip', 'voxel_labeling/data.zip',
        'voxel_labeling/trained_models.zip'
    ]
    for file in files:
        url = BASE_URL + RELEASES_TASKS[V1_IDX] + '/' + file
        localpath = os.path.join(out_dir, file)
        localdir = os.path.dirname(localpath)
        if not os.path.isdir(localdir):
          os.makedirs(localdir)
        download_file(url, localpath)
    print('Downloaded task data.')

def download_tfrecords(in_dir, out_dir):
    print('Downloading tf records (302 GB)...')
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    split_to_num_shards = {'train': 100, 'val': 25, 'test': 10}

    for folder_name in ['hires_tfrecords', 'lores_tfrecords']:
        folder_dir = '%s/%s' % (in_dir, folder_name)
        save_dir = '%s/%s' % (out_dir, folder_name)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        for split, num_shards in split_to_num_shards.items():
            for i in range(num_shards):
                file_name = '%s-%05d-of-%05d.tfrecords' % (split, i, num_shards)
                url = '%s/%s' % (folder_dir, file_name)
                localpath = '%s/%s/%s' % (out_dir, folder_name, file_name)
                download_file(url, localpath)

def download_label_map(out_dir):
    print('Downloading ScanNet ' + RELEASE_NAME + ' label mapping file...')
    files = [ LABEL_MAP_FILE ]
    for file in files:
        url = BASE_URL + RELEASE_TASKS + '/' + file
        localpath = os.path.join(out_dir, file)
        localdir = os.path.dirname(localpath)
        if not os.path.isdir(localdir):
          os.makedirs(localdir)
        download_file(url, localpath)
    print('Downloaded ScanNet ' + RELEASE_NAME + ' label mapping file.')


def main():
    parser = argparse.ArgumentParser(description='Downloads ScanNet public data release.')
    parser.add_argument('-o', '--out_dir', required=True, help='directory in which to download')
    parser.add_argument('--task_data', action='store_true', help='download task data (v1)')
    parser.add_argument('--label_map', action='store_true', help='download label map file')
    parser.add_argument('--v1', action='store_true', help='download ScanNet v1 instead of v2')
    parser.add_argument('--id', help='specific scan id to download')
    parser.add_argument('--preprocessed_frames', action='store_true', help='download preprocessed subset of ScanNet frames (' + PREPROCESSED_FRAMES_FILE[1] + ')')
    parser.add_argument('--test_frames_2d', action='store_true', help='download 2D test frames (' + TEST_FRAMES_FILE[1] + '; also included with whole dataset download)')
    parser.add_argument('--data_efficient', action='store_true', help='download data efficient task files; also included with whole dataset download)')
    parser.add_argument('--tf_semantic', action='store_true', help='download google tensorflow records for 3D segmentation / detection'