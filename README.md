# HRAnet

This codebase implements the system described in the paper:

 >3D Hierarchical Refinement and Augmentation for Unsupervised Learning of Depth and Pose from Monocular Video
 >
 >Guangming Wang, Jiquan Zhong, Shiejie Zhao, Wenhua Wu, and Hesheng Wang
 



 ## If you find our work useful in your research please consider citing our paper:
 
    @article{wang20223d,
      title={3D Hierarchical Refinement and Augmentation for Unsupervised Learning of Depth and Pose from Monocular Video},
      author={Wang, Guangming and Zhong, Jiquan and Zhao, Shijie and Wu, Wenhua and Liu, Zhe and Wang, Hesheng},
      journal={IEEE Transactions on Circuits and Systems for Video Technology},
      year={2022}
    }


## Preamble
This codebase was developed and tested with python 3.6, Pytorch 1.11.0, and CUDA 11.4 on Ubuntu 18.04. It is based on [Jia-Wang Bian's SC-SfMLearner implementation](https://github.com/JiawangBian/SC-SfMLearner-Release).



## Prerequisite

```bash
pip install -r requirements.txt
```

or install manually the following packages :

```
torch >= 1.11.0
imageio
matplotlib
scipy
argparse
tensorboardX
blessings
progressbar2
path
```

It is also advised to have python bindings for opencv for tensorboard visualizations.

You can pull the packaged docker by:
```bash
docker pull jiquanzhong/hranet:latest
```


## Datasets

See "scripts/run_prepare_data.sh".

    For KITTI Raw dataset, download the dataset using this script http://www.cvlibs.net/download.php?file=raw_data_downloader.zip) provided on the official website.

    For KITTI Odometry dataset, download the dataset with color images.

Or you can download pre-processed dataset from [Jia-Wang Bian's SC-SfMLearner implementation](https://github.com/JiawangBian/SC-SfMLearner-Release).



## Training

The "scripts" folder provides several examples for training and testing.

You can train the depth model on KITTI Raw by running
```bash
python train.py \
--name depth_pose4 --batch_size 1 \
--data /dataset/KITTI/prepared_data_completed/ \
--log-output --with-gt --epochs 200 \
--with-aug 1 --aug-padding 'zero' --augloss-mode 'quat' --aug-prob 2.0 \
--img-height 256 --img-width 832 --pose-num 2
```
or train the pose model on KITTI Odometry by running
```bash
python train.py \
--name odom_pose4 --batch_size 4 \
--data /dataset/kitti_vo_256/ \
--log-output --epoch-size 200 \
--img-height 256 --img-width 832 --pose-num 4
```
Then you can start a `tensorboard` session in this folder by
```bash
tensorboard --logdir=checkpoints/
```
and visualize the training progress by opening [https://localhost:6006](https://localhost:6006) on your browser. 



## Evaluation

You can evaluate depth on Eigen's split by running
```bash
python test_disp.py \
--dataset-dir /dataset/kitti_depth_test/color \
--resnet-layers 18 --output-dir /checkpoints/depth_pose4 \
--pretrained-dispnet /checkpoints/depth_pose4/dispnet_checkpoint.pth.tar
```
and
```bash
python eval_depth.py --dataset kitti \
 --gt_depth=/dataset/KITTI/kitti_depth_test/depth \
 --pred_depth /checkpoints/depth_pose4/predictions.npy
```
evaluate visual odometry by running
```bash
python test_vo.py --pose-num 2\
--dataset-dir /dataset/kitti_odom_test/sequences/ \
--output-dir /checkpoints/model_name \
--pretrained-dispnet /checkpoints/model_name/dispnet_checkpoint.pth.tar \
--pretrained-posenet /checkpoints/model_name/exp_pose_checkpoint.pth.tar 
```
and
```bash
python ./kitti_eval/eval_odom.py \
--result  /checkpoints/model_name  --align scale
```
    
 ## Related projects
 
 * [SC-SfMLearner-Pytorch](https://github.com/JiawangBian/SC-SfMLearner-Release)
 
 * [Kitti-Odom-Eval-Python](https://github.com/Huangying-Zhan/kitti-odom-eval)
 

 
