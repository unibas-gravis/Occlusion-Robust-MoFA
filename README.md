# Robust Model-based Face Reconstruction through Weakly-Supervised Outlier Segmentation

 Chunlu Li, [Andreas Morel-Forster](http://gravis.dmi.unibas.ch/people/ForsterA.html), [Thomas Vetter](http://gravis.dmi.unibas.ch/people/VetterT.html), [Bernhard Egger*](https://eggerbernhard.ch/), and [Adam Kortylewski*](https://generativevision.mpi-inf.mpg.de/)

[pdf](https://arxiv.org/pdf/2106.09614.pdf) | [video](https://youtu.be/7nKbNmupViM)


This work enables a model-based face autoencoder to segment occlusions accurately for 3D face reconstruction and provides state-of-the-art occlusion segmentation results and the face reconstruction is robust to occlusions. It requires only weak supervision for the face reconstruction subnetwork and can be trained end-to-end efficiently. The effectiveness of this method is verified on the Celeb A HQ dataset, the AR dataset, and the NoW Challenge.


#### ● [Update 20230331] [Docker image](https://hub.docker.com/r/chunluli/focus) with trained model available now!

#### ● [Update 20230321] Accepted by CVPR 2023!
  1. Docker with pre-trained model coming soon.
  
#### ● [Update 20210315] Reached the SOTA on the NoW Challenge!
  1. The ArcFace for perceptual-level loss.

  2. Better tuned hyper-parameters for higher reconstruction accuracy.

  3. Test and evaluation code released. 3D shape (.obj mesh), rendered faces, and estimated masks available. Evaluation indices (accuracy, precision, F1 socre, and recall rate) available. 


## Features

### ● Accurate Occlusion Segmentation

This method provides reliable occlusion segmentation masks and the training of the segmentation network **does not require any additional supervision**.

<p align="center"> 
<img src="https://github.com/unibas-gravis/Occlusion-Robust-MoFA/blob/main/visual_results.jpg">
</p>


### ● Occlusion-robust 3D Reconstruction

This method produces accurate 3D face model fitting results which are robust to occlusions.

##### [New!] Our method, named 'FOCUS' (Face-autoencoder and OCclUsion Segmentation), reaches the SOTA on the NoW Challenge!

The results of the state-of-the-art methods on the NoW face benchmark is as follows:
|Rank|Method|Median(mm)    | Mean(mm) | Std(mm) |
|:----:|:-----------:|:-----------:|:-----------:|:-----------:|
| **1.** | **FOCUS (Ours)**|**1.04**|**1.30**|**1.10**|
| 2. | [DECA\[Feng et al., SIGGRAPH 2021\]](https://github.com/YadiraF/DECA)|1.09|1.38|1.18|
| 3. | [Deep3DFace PyTorch [Deng et al., CVPRW 2019]](https://github.com/sicxu/Deep3DFaceRecon_pytorch)|1.11|1.41|1.21|
| 4. | 	[RingNet [Sanyal et al., CVPR 2019]](https://github.com/soubhiksanyal/RingNet) | 1.21 | 1.53 | 1.31 |
| 5. | [Deep3DFace [Deng et al., CVPRW 2019]](https://github.com/microsoft/Deep3DFaceReconstruction) | 1.23 | 1.54 | 1.29 |
| 6. | [3DDFA-V2 [Guo et al., ECCV 2020]](https://github.com/cleardusk/3DDFA_V2) | 1.23 | 1.57 | 1.39 |
| 7. | [MGCNet [Shang et al., ECCV 2020]](https://github.com/jiaxiangshang/MGCNet) | 1.31 | 1.87 | 2.63 |
| 8. | [PRNet [Feng et al., ECCV 2018]](https://github.com/YadiraF/PRNet) | 1.50 | 1.98 | 1.88 |
| 9. | [3DMM-CNN [Tran et al., CVPR 2017]](https://github.com/anhttran/3dmm_cnn) | 1.84 | 2.33 | 2.05 |

For more details about the evaluation, check [Now Challenge](https://ringnet.is.tue.mpg.de/challenge.html) website.


### ● Easy to implement

This method follows a step-wise manner and is easy to implement.

## Getting Started

To train and/or test this work, you need to:

   1. [Prepare the data](https://github.com/unibas-gravis/Occlusion-Robust-MoFA#-data-preparation)
    
   2. [Download 3DMM](https://github.com/unibas-gravis/Occlusion-Robust-MoFA#-download-3dmm)
   
   3. [Install Arcface](https://github.com/unibas-gravis/Occlusion-Robust-MoFA#-install-arcface-for-perceptual-loss)
    
   4. [Install Dependencies](https://github.com/unibas-gravis/Occlusion-Robust-MoFA#-install-dependencies)
    
   5. [Train step-by-step](https://github.com/unibas-gravis/Occlusion-Robust-MoFA#-step-wise-training)
   
   6. [Test](https://github.com/unibas-gravis/Occlusion-Robust-MoFA#-testing)

### ● Data Preparation
  
  
1. Prepare .csv files for the training set, validation set, and testing set.

    The .csv files should contain rows of [filename + landmark coordinates].
    
    We recommend using the 68 2D landmarks detected by [2D-and-3D-face-alignment](https://github.com/1adrianb/2D-and-3D-face-alignment).

2. To evaluate the accuracy of the estimated masks, ground truth occlusion segmentation masks are required. Please name the target image as 'image_name.jpg' and ground truth masks as 'image_name_visible_skin_mask.png'.

   The image directory should follow the structure below:
    
		./image_root
		├── Dataset                     # Database folder containing the train set, validation set, and test set.
		    ├──1.jpg                    # Target image
		    ├──1_visible_skin_mask.png  # GT masks for testing. (optional for training)
		    └──...
		├── train_landmarks.csv      # .csv file for the train set.
		├── test_landmarks.csv       # .csv file for the test set.
		├── val_landmarks.csv        # .csv file for the validation set.
		└── all_landmarks.csv        # .csv file for the whole dataset. (optional)

  

 


### ● Download 3DMM

  1. Our implementation employs the [BFM 2017](https://faces.dmi.unibas.ch/bfm/bfm2017.html). Please copy 'model2017-1_bfm_nomouth.h5' to './basel_3DMM'.


### ● Install ArcFace for Perceptual Loss
  We depend on [ArcFace](https://github.com/deepinsight/insightface/tree/master/recognition/arcface_torch) to compute the perceptual features for the target images and the rendered image. 
  1. Download the trained [model](https://onedrive.live.com/?authkey=%21AFZjr283nwZHqbA&id=4A83B6B633B029CC%215583&cid=4A83B6B633B029CC).

  2. Place ms1mv3_arcface_r50_fp16.zip and backbone.pth under ./Occlusion_Robust_MoFA/models/.
  
  3. To install the ArcFace, please run the following code:
  
    cd ./Occlusion_Robust_MoFA
    git clone https://github.com/deepinsight/insightface.git
    cp -r ./insightface/recognition/arcface_torch/* ./models/
    
  4. Overwrite './models/backbones/iresnet.py' with the file in our repository.
  
  The structure of the directory 'models' should be:
  
  		./models
		├── ms1mv3_arcface_r50_fp16
		    ├──backbone.pth
		    └──...                       # Trained model downloaded.
		├── backbones
		    ├──*iresnet.py               # Overwritten by our code.
		    └──...
		└── ...                          # files/directories downloaded from ArcFace repo.

### ● Install Dependencies

  We recommend using anaconda or miniconda to create virtue environment and install the packages. You can set up the environment with the following commands:

    conda create -n FOCUS python=3.6
    conda activate FOCUS
    pip install -r requirements.txt
    

### ● Step-wise Training

To train the proposed network, please follow the steps:
  1. Enter the directory

	cd ./Occlusion_Robust_MoFA

  2. Unsupervised Initialization
    
	python Step1_Pretrain_MoFA.py --img_path ./image_root/Dataset
    
  3. Generate UNet Training Set

	python Step2_UNet_trainset_generation.py --img_path ./image_root/Dataset

  4. Pretrain Unet

	python Step3_Pretrain_Unet.py

  5. Joint Segmentation and Reconstruction

	python Step4_UNet_MoFA_EM.py --img_path ./image_root/Dataset

  6. Test-time adaptation (Optional) 
  
	 To  bridge the domain gap between training and testing data to reach higher performance on the test dataset, test-time adaptation is available with the following command: 
	
	python Step4_UNet_MoFA_EM.py --img_path ./image_root/Dataset_adapt --pretrained_model iteration_num

[//]: # (TODO: release model)

### ● Testing
   To test the model saved as './MoFA_UNet_Save/model-path/model-name', use the command below:
   
	python Demo.py --img_path ./image_root/Dataset --pretrained_model_test ./MoFA_UNet_Save/model-path/model-name.model --test_mode pipeline_name --test_path test_dataset_root --save_path save_path --landmark_list_name landmark_filename_optional.csv

## Docker Image

### ● Differences
  1. .csv files are no longer required in the docker version. Instead, the landmarks are automatically detected.
  2. Fixed the naming of some variables.
  3. Misfit prior is also included in the docker image.

### ● Getting started

  1. Pull.

    sudo docker pull chunluli/focus:1.2

  2. Run a container with your data directory /DataDir mounted.

    docker run -v /DataDir:/FOCUS/data -itd chunluli/focus:1.2 /bin/bash 
    
    docker attach containerID

  3. Run the following command to see how to use the codes:

    python show_instructions.py
   
  More information can be found in [dockerhub](https://hub.docker.com/r/chunluli/focus).


## Citation

Please cite the following papers if this model helps your research:

    @article{li2021fit,
    title={To fit or not to fit: Model-based Face Reconstruction and Occlusion Segmentation from Weak Supervision},
    author={Li, Chunlu and Morel-Forster, Andreas and Vetter, Thomas and Egger, Bernhard and Kortylewski, Adam},
    journal={arXiv preprint arXiv:2106.09614},
    year={2021}}
    
This code is built on top of the MoFA re-implementation from Tatsuro Koizumi and the data processing is on top of the Deep3D. If you establish your own work based on our work, please also cite the following papers:

    @inproceedings{koizumi2020look,
      title={“Look Ma, no landmarks!”--Unsupervised, model-based dense face alignment},
      author={Koizumi, Tatsuro and Smith, William AP},
      booktitle={European Conference on Computer Vision},
      pages={690--706},
      year={2020},
      organization={Springer}
    }
    
    @inproceedings{deng2019accurate,
    title={Accurate 3D Face Reconstruction with Weakly-Supervised Learning: From Single Image to Image Set},
    author={Yu Deng and Jiaolong Yang and Sicheng Xu and Dong Chen and Yunde Jia and Xin Tong},
    booktitle={IEEE Computer Vision and Pattern Recognition Workshops},
    year={2019}
    }
