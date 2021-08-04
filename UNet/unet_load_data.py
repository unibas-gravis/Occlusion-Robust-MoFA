#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 19 14:55:13 2021

@author: li0005
"""


import torch
import cv2 as cv
import csv
import torchvision.transforms.functional as tf
import numpy as np
import os.path



class UNetDataset(torch.utils.data.Dataset):

    def __init__(self,device,root,train,height,width,scale,landmark_file=False):
        super(UNetDataset,self)

        self.train = train

        if self.train:
            landmark_filename = root+'../train_landmark_align_celeba.csv'
        else:
            landmark_filename = root+'../test_landmark_align_celeba.csv'

        
        self.landmark_list = list(csv.reader(open(landmark_filename),delimiter=','))
        self.num = len(self.landmark_list)
        self.root = root
        self.train = train
        self.device = device
        self.scale = scale
        self.width = width
        self.height = height
              
        
    def __len__(self):
        return self.num
    
    def __getitem__(self,index):
        
        org_filename = self.root+self.landmark_list[index][0].replace('.jpg','_org.jpg')

        if os.path.exists(org_filename):
            org_image = cv.imread(org_filename)
            width_img = org_image.shape[1]
        else:
            print('Org File not exist: \n'+org_filename)           
            
        rastered_filename = org_filename.replace('_org.jpg','_raster.jpg')
        rastered_image = cv.imread(rastered_filename)
            
        mask_filename = org_filename.replace('_org.jpg','_mask.jpg')
        
        if os.path.exists(mask_filename):
            mask_image = cv.imread(mask_filename)[:,:,0]
        else:
            print('mask File not exist: \n'+mask_filename)
            
            
        width_img = org_image.shape[1]
        height_img = org_image.shape[0]
        size_img = min(width_img,height_img)
        size_scaled = int(size_img*self.scale)


        img_input = np.concatenate((tf.to_pil_image(org_image), tf.to_pil_image(rastered_image)),axis=2)
        IMAGE_INPUT = tf.to_tensor(img_input)
        IMAGE_INPUT = torch.flip(IMAGE_INPUT, [0]).to(self.device)
        
        
        MASK_GT = tf.to_tensor(tf.center_crop(tf.resize(tf.to_pil_image(mask_image),size_scaled),(self.height,self.width)))
        MASK_GT = torch.flip(MASK_GT, [0]).to(self.device)
  

        
        return IMAGE_INPUT, MASK_GT,org_image,rastered_image 
    
    
    
    
    
    
    