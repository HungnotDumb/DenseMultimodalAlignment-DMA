# Multi-view Feature Fusion

## Overview

Here we provide instructions for multi-view feature fusion on different dataset, including ScanNet, Matterport3D, nuScenes, and Replica. This corresponds to **Section 3.1** in our [paper](https://arxiv.org/abs/2211.15654).

**Note**: For now we provide only codes for multiview fusion with the **[OpenSeg](https://github.com/tensorflow/tpu/tree/master/models/official/detection/projects/openseg)** model. However, you can easily adapt our code for other per-pixel feature extractor like [LSeg](https://github.com/isl-org/lang-seg) or [OVSeg](https://github.com/facebookresearch/ov-seg).


## Prerequisites

### Data preprocessing
Follow [this instruction](../preprocess/README.md) to obtain the processed 2D and 3D data.
- **3D**: Point clouds in the pytorch format `.pth`
- **2D**: RGB-D images with their intrinsic and extrinsic parameters

### Envinroment
You can simply activate the `openscene` conda environment, or alternatively, make sure the following package installed:
- `torch`
- `tensorflow v2` (for OpenSeg feature extraction)
- `numpy`
- `imageio`
- `tqdm`

To use **OpenSeg** as the