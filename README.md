# To fit or not to fit: Model-based Face Reconstruction and Occlusion Segmentation from Weak Supervision 

This is a pytorch implementation of the following paper:

[To fit or not to fit: Model-based Face Reconstruction and Occlusion Segmentation from Weak Supervision](https://arxiv.org/pdf/2106.09614.pdf)


This work enables a model-based face autoencoder to segment occlusions accurately for 3D face reconstruction and provides state-of-the-art occlusion segmentation results and the face reconstruction is robust to occlussions. It requires only weak supervision for the face reconstruction subnetwork and can be trained end-to-end efficiently. The effectiveness of this method is verified on the Celeb A HQ dataset and the AR dataset.


## Features

### Accurate Occlusion Segmentation from Weak Supervision

This method provides reliable occlusion segmentation masks and the training of the segmentation network does not require any additional supervision.

<p align="center"> 
<img src="https://github.com/ChunLLee/Occlusion_Robust_MoFA/blob/main/samples_segmentation.png">
</p>


### Occlusion-robust Model Fitting

This method produces accurate 3D face model fitting results which are robust to occlusions.

<p align="center"> 
<img src="https://github.com/ChunLLee/Occlusion_Robust_MoFA/blob/main/samples_fitting.png">
</p>


### Easy to implement

This method follows a step-wise manner and is easy to implement.

## Getting Started

To train or test this work, you need to:

### ● Data Preparation

1. Conduct landmark detection.

    We recommend to use center landmarks detected by [Dlib](https://github.com/davisking/dlib) and the landmarks on the 3D contours from [2D-and-3D-face-alignment](https://github.com/1adrianb/2D-and-3D-face-alignment).
  
2. Prepare .csv files for the training set, validation set, and the testing set.

    The .csv files should contain rows of [filename + landmark coordinates] and the image directory should follow the structure below:
    
		./image_root
		├── Dataset                     # Database folder containing the train set, validation set and test set.
		├── train_landmarks_3D.csv      # .csv file for the train set.
		├── test_landmarks_3D.csv       # .csv file for the test set.
		├── val_landmarks_3D.csv        # .csv file for the validation set.
		└── all_landmarks_3D.csv        # .csv file for the whole dataset.

  

  
3. To evaluate the accuracy of the estimated masks, you need to prepare the ground truth occlusion segmentation masks.


### ● Download 3DMM

  Our implementation employs the [BFM 2017](https://faces.dmi.unibas.ch/bfm/bfm2017.html). Please copy 'model2017-1_bfm_nomouth.h5' to './basel_3DMM'.

### ● Install Dependencies

  We recommend to use anaconda or miniconda to create virtue environment and install the packages. You can set up the environment with the following commands:

    conda create -n env_name python=3.6
    conda activate env_name
    pip install -r requirements.txt
    

### ● Step-wise Training

To train the proposed network, please follow the steps:
  1. Enter the directory

	cd ./Occlusion_Robust_MoFA

  2. Pretrain MoFA
    
	python Step1_Pretrain_MoFA.py --img_path ./image_root
    
  3. Generate UNet Training Set

	python Step2_UNet_trainset_generation.py --img_path ./image_root

  4. Pretrain Unet

	python Step3_Pretrain_Unet.py

  5. Joint Segmentation and Reconstruction

	python Step4_UNet_MoFA_EM.py --img_path ./image_root

[//]: # (### ● Test with Pre-trained Model To test the proposed method, please download the [pretrained model]http://GoogleDrive//TOBERELEASED and save the model at './Pretrained_Model' and then conduct 'demo.py'.)

## Citation

Please cite the following papers if this model helps your research:

    @article{li2021fit,
      title={To fit or not to fit: Model-based Face Reconstruction and Occlusion Segmentation from Weak Supervision},
      author={Li, Chunlu and Morel-Forster, Andreas and Vetter, Thomas and Egger, Bernhard and Kortylewski, Adam},
      journal={arXiv preprint arXiv:2106.09614},
      year={2021}
    }
    
This code is built on top of the MoFA re-implementation from Tatsuro Koizumi. If you establish your own work based on our work, please also cite the following paper:

    @inproceedings{koizumi2020look,
      title={“Look Ma, no landmarks!”--Unsupervised, model-based dense face alignment},
      author={Koizumi, Tatsuro and Smith, William AP},
      booktitle={European Conference on Computer Vision},
      pages={690--706},
      year={2020},
      organization={Springer}
    }
