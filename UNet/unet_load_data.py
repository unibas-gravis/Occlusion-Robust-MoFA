import torch
import cv2
import csv
import torchvision.transforms.functional as tf
from torchvision.transforms import Normalize
import numpy as np
import pickle
import random
from PIL import Image
from util.load_mats import load_lm3d
import os

import torchvision.transforms as transforms
from util.preprocess import *
def unpickle(file):
    
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict
def get_transform(grayscale=False):
    transform_list = []
    if grayscale:
        transform_list.append(transforms.Grayscale(1))
    transform_list += [transforms.ToTensor()]
    return transforms.Compose(transform_list)

def get_affine_mat(preprocess='shift_scale_rot_flip', size=(224,224)):
    shift_x, shift_y, scale, rot_angle, flip = 0., 0., 1., 0., False
    w, h = size
    scale_delta=0.1
    rot_angle=10

    if 'shift' in preprocess:
        shift_pixs = int(10)
        shift_x = random.randint(-shift_pixs, shift_pixs)
        shift_y = random.randint(-shift_pixs, shift_pixs)
    if 'scale' in preprocess:
        scale = 1 + scale_delta * (2 * random.random() - 1)
    if 'rot' in preprocess:
        rot_angle = rot_angle * (2 * random.random() - 1)
        rot_rad = -rot_angle * np.pi/180
    if 'flip' in preprocess:
        flip = random.random() > 0.5

    shift_to_origin = np.array([1, 0, -w//2, 0, 1, -h//2, 0, 0, 1]).reshape([3, 3])
    flip_mat = np.array([-1 if flip else 1, 0, 0, 0, 1, 0, 0, 0, 1]).reshape([3, 3])
    shift_mat = np.array([1, 0, shift_x, 0, 1, shift_y, 0, 0, 1]).reshape([3, 3])
    rot_mat = np.array([np.cos(rot_rad), np.sin(rot_rad), 0, -np.sin(rot_rad), np.cos(rot_rad), 0, 0, 0, 1]).reshape([3, 3])
    scale_mat = np.array([scale, 0, 0, 0, scale, 0, 0, 0, 1]).reshape([3, 3])
    shift_to_center = np.array([1, 0, w//2, 0, 1, h//2, 0, 0, 1]).reshape([3, 3])
    
    affine = shift_to_center @ scale_mat @ rot_mat @ shift_mat @ flip_mat @ shift_to_origin    
    affine_inv = np.linalg.inv(affine)
    return affine, affine_inv, flip

def apply_img_affine(img, affine_inv, method=Image.BICUBIC):
    return img.transform(img.size, Image.AFFINE, data=affine_inv.flatten()[:6], resample=Image.BICUBIC)

def apply_lm_affine(landmark, affine, flip, size):
    _, h = size
    lm = landmark.copy()
    lm[:, 1] = h - 1 - lm[:, 1]
    lm = np.concatenate((lm, np.ones([lm.shape[0], 1])), -1)
    lm = lm @ np.transpose(affine)
    lm[:, :2] = lm[:, :2] / lm[:, 2:]
    lm = lm[:, :2]
    lm[:, 1] = h - 1 - lm[:, 1]
    if flip:
        lm_ = lm.copy()
        lm_[:17] = lm[16::-1]
        lm_[17:22] = lm[26:21:-1]
        lm_[22:27] = lm[21:16:-1]
        lm_[31:36] = lm[35:30:-1]
        lm_[36:40] = lm[45:41:-1]
        lm_[40:42] = lm[47:45:-1]
        lm_[42:46] = lm[39:35:-1]
        lm_[46:48] = lm[41:39:-1]
        lm_[48:55] = lm[54:47:-1]
        lm_[55:60] = lm[59:54:-1]
        lm_[60:65] = lm[64:59:-1]
        lm_[65:68] = lm[67:64:-1]
        lm = lm_
    return lm
def parse_label(label):
    return torch.tensor(np.array(label).astype(np.float32))
class UNetDataset(torch.utils.data.Dataset):

    def __init__(self,device,root,train,landmark_file,height,width,scale=1,test_mode = False,is_use_aug=True,bfm_folder='./BFM'):
        super(UNetDataset,self)
        self.train = train
        self.test_mode = test_mode

        if self.train:
            landmark_filename = root+'../train_landmarks.csv'
        else:
            landmark_filename = root+'../val_landmarks.csv'
        if self.test_mode:
            self.train = False
            if landmark_file:
                landmark_filename = landmark_file
            else:
                landmark_filename = root+'../test_landmarks.csv'
        current_path = os.getcwd()
        self.root = current_path +'../MoFA_UNet_Save/UNet_trainset/' 
        print(landmark_filename)
        f=open(landmark_filename)
        f_csv=csv.reader(f,delimiter=',')
        self.landmark_list = list(f_csv)
        random.shuffle(self.landmark_list)
        self.num = len(self.landmark_list)
        self.use_aug = is_use_aug
        self.device = device
        self.scale = scale
        self.width = width
        self.height = height
        self.lm3d_std = load_lm3d(bfm_folder)
        

        
    def __len__(self):
        return self.num

    def __getitem__(self,index):

        try:
            filename = self.root+self.landmark_list[index][0].replace('.jpg','_org.jpg')
        except:
            pass
        if not os.path.exists(filename):
            print('Image does not exist: ' + filename)
        landmark_cpu = [int(x) for x in self.landmark_list[index][1:]]



        try:
            image = cv2.imread(filename)
            width_img = image.shape[1]
        except:
            print(filename)
        raw_img = Image.open(filename).convert('RGB')
        #raw_msk = Image.open(msk_path).convert('RGB')
        _, H = raw_img.size
        raw_lm = np.reshape(np.asarray(landmark_cpu),(-1,2)).astype(np.float32)
        raw_lm = np.stack([raw_lm[:,0],H-raw_lm[:,1]],1)
        
        
        
        _, img, lm, msk = align_img(raw_img, raw_lm, self.lm3d_std, mask=None)
            
        skin_vis_mask_path = filename.replace('.jpg','_mask.jpg')
        rastered_filename = filename.replace('.jpg','_raster.jpg')
        if os.path.exists(skin_vis_mask_path):
                
            _,num_temp = os.path.split(filename)
            num_temp = int(num_temp[:-8])
               
            raw_gtmask = Image.open(skin_vis_mask_path).convert('RGB')
            _, gt_mask, _, _ = align_img(raw_gtmask, raw_lm, self.lm3d_std, mask=None)
                
            raw_raster = Image.open( rastered_filename).convert('RGB')
            _, gt_raster, _, _ = align_img(raw_raster, raw_lm, self.lm3d_std, mask=None)
        else:
            print('!Mask does not exist: '+filename)
            index+=1
            self.__getitem__(index+1)
        
        # Data augmentation is only for training
        aug_flag = self.use_aug and self.train

        if aug_flag :
            msk=gt_mask
            raster=gt_raster
            img, lm, gt_mask , gt_raster= self._augmentation(img, lm, msk,raster)
        _, H = img.size
        transform = get_transform()
        img_tensor = transform(img).to(self.device)

        gt_raster_tensor = transform(gt_raster)[:1, ...].to(self.device)
        gt_mask_tensor = transform(gt_mask)[:1, ...].to(self.device)
        
        img_input = np.concatenate((img, gt_raster),axis=2)
        IMAGE_INPUT = tf.to_tensor(img_input)
        IMAGE_INPUT = torch.flip(IMAGE_INPUT, [0]).to(self.device)
        return IMAGE_INPUT, gt_mask_tensor,img_tensor,gt_raster_tensor

            
    def _augmentation(self, img, lm,  msk=None,raster=None):
        affine, affine_inv, flip = get_affine_mat()
        img = apply_img_affine(img, affine_inv)
        lm = apply_lm_affine(lm, affine, flip, img.size)
        if msk is not None:
            msk = apply_img_affine(msk, affine_inv, method=Image.BILINEAR)
        if raster is not None:
            aug_raster = apply_img_affine(raster, affine_inv, method=Image.BILINEAR)
        return img, lm, msk,aug_raster
