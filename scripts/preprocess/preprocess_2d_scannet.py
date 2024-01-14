# pre-process ScanNet 2D data
# code adapted from https://github.com/angeladai/3DMV/blob/master/prepare_data/prepare_2d_data.py
#
# note: depends on the sens file reader from ScanNet:
#       https://github.com/ScanNet/ScanNet/blob/master/SensReader/python/SensorData.py
# which is renamed to scannet_sensordata.py under this directory

# Example usage:
#    python prepare_2d_scannet.py --sc