echo "This script downloads multi-view fused features used in the OpenScene project."
echo "Choose from the following options:"
echo "0 - ScanNet - Multi-view fused OpenSeg features, train/val (234.8G)"
echo "1 - ScanNet - Multi-view fused LSeg features, train/val (175.8G)"
echo "2 - Matterport - Multi-view fused OpenSeg features, train/val (198.3G)"
echo "3 - Matterport - Multi-view fused OpenSeg features, test set (66.7G)"
echo "4 - Replica - Multi-view fused OpenSeg features (9.0G)"
echo "5 - Matterport - Multi-view fused LSeg features (coming)"
echo "6 - nuScenes - Multi-view fused OpenSeg features (coming)"
echo "7 - nuScenes - Multi-view fused LSeg features (coming)"
read -p "Enter dataset ID you want to download: " ds_id


if [ $ds_id == 0 ]
then
    echo "You chose 0: 