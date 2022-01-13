# V2V-PoseNet-PyTorch
This is a reimplementation of the PyTorch implementation of [V2V-PoseNet: Voxel-to-Voxel Prediction Network for Accurate 3D Hand and Human Pose Estimation from a Single Depth Map](https://arxiv.org/abs/1711.07399), created by [@dragonbook](https://github.com/dragonbook), which is largely based on the author's [torch7 implementation](https://github.com/mks0601/V2V-PoseNet_RELEASE).

This repository provides:
* V2V-PoseNet core modules (model, voxelization, ..)
* An trained model on MSRA hand pose dataset, with about a ~11mm mean error.

## Requirements
Tested on a Windows 11 AMD 5950x Nvidia 3090 machine running:
* Python 3.9.9
* numpy 1.22.0
* open3d 0.14.1.0
* torch 1.10+cu113 (pytorch)

## Optional Requirements
If you wish to convert the ITOP dataset for use in the model you will need the following:
* h5py 3.6.0

## How to install
1. Clone the repo
2. Open a Python terminal in the root directory of the repo
3. Run the following to install the dependencies  
   ```python3 install_requirements.py```
4. Install PyTorch (with CUDA) the install link for this on Windows 10/11 with a modern Nvidia GPU is as follows:  
   ```pip3 install torch==1.10.1+cu113 torchvision==0.11.2+cu113 torchaudio===0.10.1+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html```  
   For other installs use the config tool at [pytorch.org](https://pytorch.org/) to download it

## MSRA Hand Gesture dataset
### Downloading the dataset
The dataset and the centers can be found at:
* [MSRA hand dataset](https://jimmysuen.github.io/) - and extract to `/datasets/cvpr15_MSRAHandGestureDB`
* [Estimated centers](https://cv.snu.ac.kr/research/V2V-PoseNet/MSRA/center/center.tar.gz)  - and extract to `/datasets/msra_center`

The dataset is described in the paper [Cascaded Hand Pose Regression, Xiao Sun, Yichen Wei, Shuang Liang, Xiaoou Tang and Jian Sun, CVPR 2015](https://www.cv-foundation.org/openaccess/content_cvpr_2015/papers/Sun_Cascaded_Hand_Pose_2015_CVPR_paper.pdf).  
The estimated centers are provided by the [original author's implementation](https://github.com/mks0601/V2V-PoseNet_RELEASE).  

For simplicity the centers are currently included in the repo.  

### Training a model on the dataset
1. Open a python terminal in the root directory of the repo
2. Run the following `python3 experiments/msra-subject3/main.py`
3. Let it run, after 15 epochs it will output to `/output/TIMESTAMP/`  
   The output contains
     - `/checkpoint/`  
       Contains the checkpoint files from each epoch
     - `model.pt`  
       The exported model
     - `fit_res.txt` & `test_res.txt`  
       Used by the visualizer (see below)

### Pre-trained model
A pre-trained model is included in `/output/cvpr15_MSRAHandGestureDB/model.pt`


## ITOP Dataset
### Downloading the dataset
The dataset and the centers can be found at:
* [ITOP dataset](https://zenodo.org/record/3932973) - and extract to `/datasets/ITOP`
* [Estimated centers](https://drive.google.com/drive/folders/1-v-VN-eztzoztfHcLt_Y8o5zfRosJ6jt)  - and extract to `/datasets/ITOP_side_center`

The dataset is described in the paper [Towards Viewpoint Invariant 3D Human Pose Estimation, Albert Haque, Boya Peng, Zelun Luo, Alexandre Alahi, Serena Yeung, Li Fei-Fei, CVPR 2016](https://arxiv.org/abs/1603.07076).  
The estimated centers are provided by the [original author's implementation](https://github.com/mks0601/V2V-PoseNet_RELEASE).

For simplicity the centers are currently included in the repo.  

Your final `/dataset/` folder should look like:   
```
/datasets/ITOP 
/datasets/ITOP/ITOP_side_test_labels.h5
/datasets/ITOP/ITOP_side_test_point_cloud.h5
/datasets/ITOP/ITOP_side_train_labels.h5
/datasets/ITOP/ITOP_side_train_point_cloud.h5

/datasets/ITOP_side_center
/datasets/ITOP_side_center/center_test.txt
/datasets/ITOP_side_center/center_train.txt
```

### Pre-preparing the dataset
As the dataset is very large we preprocess it all into smaller files for each frame.  
A helper file is provided, which is run using the following:  
`python3 datasets/itop_side_preprocess.py`  
This will generate a ~10GB directory at `/datasets/ITOP_side_processed`

### Training a model on the dataset
1. Open a python terminal in the root directory of the repo
2. Run the following `python3 experiments/itop_side/main.py`
3. Let it run, after 15 epochs it will output to `/output/TIMESTAMP/`  
   The output contains
     - `/checkpoint/`  
       Contains the checkpoint files from each epoch
     - `model.pt`  
       The exported model
     - `fit_res.txt` & `test_res.txt`  
       Used by the visualizer


### Pre-trained model
A pre-trained model is included in `/output/ITOP_side/model.pt`


## Model accuracy
To see how well a model is trained, run the following:  
`python3 output/OUTPUT_FOLDER/accuracy_graph.py`  
for the 2 pre-trained models the command is as follows:
- MSRA  
  `python3 output/cvpr15_MSRAHandGestureDB/accuracy_graph.py`  
- ITOP side  
  `python3 output/ITOP_side/accuracy_graph.py`  

This will display 2 graphs which can be used to assess the accuracy of the model.  


## Example of a model in use
Some demonstration code of the ITOP side model is provided, and can be run using the following from the root directory of the repo:  
`python3 example/itop.py`  
This code pulls the test data from the ITOP side dataset and runs the model on them.

## Original authors
Moon, Gyeongsik, Ju Yong Chang, and Kyoung Mu Lee. **"V2V-PoseNet: Voxel-to-Voxel Prediction Network for Accurate 3D Hand and Human Pose Estimation from a Single Depth Map."** <i>CVPR 2018. </i> [[arXiv](https://arxiv.org/abs/1711.07399)]
  
  ```
@InProceedings{Moon_2018_CVPR_V2V-PoseNet,
  author = {Moon, Gyeongsik and Chang, Juyong and Lee, Kyoung Mu},
  title = {V2V-PoseNet: Voxel-to-Voxel Prediction Network for Accurate 3D Hand and Human Pose Estimation from a Single Depth Map},
  booktitle = {The IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
  year = {2018}
}
```
