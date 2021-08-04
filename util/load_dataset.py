

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 14 15:18:24 2021

@author: Gravis
"""

import torch
import cv2
import csv
import torchvision.transforms.functional as tf
from torchvision.transforms import Normalize
import numpy as np
import pickle
from pathlib import Path
import os.path
import random

def unpickle(file):
    
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

class CelebDataset(torch.utils.data.Dataset):

    def __init__(self,device,root,train,height,width,scale,landmark_file=False,test_mode = False):
        super(CelebDataset,self)
        self.test_mode = test_mode
        self.train = train
        if self.train:
            landmark_filename = root+'../train_landmarks_3D.csv'
        else:
            landmark_filename = root+'../val_landmarks_3D.csv'
        if self.test_mode:
            if landmark_file:
                landmark_filename = landmark_file
            else:
                landmark_filename = root+'../test_landmarks_3D.csv'
        self.landmark_list = list(csv.reader(open(landmark_filename),delimiter=','))
        self.num = len(self.landmark_list)
        

        self.root = root
        self.device = device
        self.scale = scale
        self.width = width
        self.height = height
        
        
    def __len__(self):
        return self.num
    
    def shuffle(self):
        random.shuffle(self.landmark_list)

    
    def __getitem__(self,index):
        
        
        filename = self.root+self.landmark_list[index][0]
        landmark_cpu = [int(x) for x in self.landmark_list[index][1:]]
         

        if os.path.exists(filename):
            image = cv2.imread(filename)
            image=cv2.resize(image,(224,224))
        else:
            print(filename)
           
        image_org = image.copy()
        width_img = image.shape[1]
        height_img = image.shape[0]
        size_img = min(width_img,height_img)
        size_scaled = int(size_img*self.scale)
        
        center_img = torch.Tensor([[width_img/2.0],[height_img/2.0]])
        center_scaled = torch.Tensor([[self.width/2.0],[self.height/2.0]])
        
        landmark = torch.Tensor(landmark_cpu).reshape(-1,2).transpose(0,1)
        landmark = (landmark-center_img) * float(size_scaled)/float(size_img) + center_scaled
        landmark = landmark.to(self.device)
        skin_vis_mask_path = filename.replace('.jpg','_visible_skin_mask.png')
        
        
        if self.test_mode:
            image_ORG = tf.to_tensor(tf.center_crop(tf.resize(tf.to_pil_image(image_org),size_scaled),(self.height,self.width)))
            image_ORG = torch.flip(image_ORG, [0]).to(self.device)
            if os.path.exists(skin_vis_mask_path):
                image_mask = cv2.imread(skin_vis_mask_path)
                image_mask=cv2.resize(image_mask,(224,224))
                valid_mask = tf.to_tensor(tf.center_crop(tf.resize(tf.to_pil_image(image_mask),size_scaled),(self.height,self.width)))
                valid_mask = torch.flip(valid_mask, [0]).to(self.device)
            else:
                valid_mask = False
            return image_ORG,filename, valid_mask

        
        image_input = image
        image_input = tf.to_tensor(tf.center_crop(tf.resize(tf.to_pil_image(image_input),size_scaled),(self.height,self.width)))
        image_input = torch.flip(image_input, [0]).to(self.device)
        
        
        
        return image_input, landmark
        

    
    
    
    
    
    
    
