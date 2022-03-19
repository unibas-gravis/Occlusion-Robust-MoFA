#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 30 15:36:21 2021

@author: li0005
"""
#import util.random_synthesis as ransyn
import torch
import numpy as np

def neighbor_unet_loss(images,est_mask,raster_image,neighbour_width=5,neighbour_height=5):
    _,c,w,h = images.shape
    margin_width = int(np.floor(neighbour_width/2))
    margin_height = int(np.floor(neighbour_height/2))
    cropped_rendered_img =(raster_image* est_mask)[:,:,margin_width:(w-margin_width),margin_height:(h-margin_height )]
    cropped_target_img = (images* est_mask)
    
    start=True
    for i in range(-1 * margin_width,margin_width+1):
        for j in range(-1 * margin_height,margin_height+1):
            left = margin_width+i
            top = margin_height+j
            right = w-margin_width+i
            bottom = h-margin_height +j
            if start:
                lowest = torch.norm(cropped_rendered_img-cropped_target_img[:,:,left:right,top:bottom],2,1)
                start=False
            else:
                crop_temp = torch.norm(cropped_rendered_img-cropped_target_img[:,:,left:right,top:bottom],2,1)
                lowest = torch.minimum(crop_temp,lowest)
    return torch.mean(lowest)

