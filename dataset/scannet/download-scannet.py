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
    out_dir = os.path.dirname(out_