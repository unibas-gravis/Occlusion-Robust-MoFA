#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 30 15:36:21 2021

@author: Gravis
"""

import torch




def neighbor_unet_loss(images, est_mask, raster_image):
    _,c,w,h = images.shape
    cropped_rendered_img =(raster_image* est_mask)[:,:,1:(w-1),1:(h-1)]
    cropped_target_img = (images* est_mask)
    lt = torch.norm(cropped_rendered_img-cropped_target_img[:,:,0:(w-2),0:(h-2)],2,1)
    mt = torch.norm(cropped_rendered_img-cropped_target_img[:,:,1:(w-1),0:(h-2)],2,1)
    a= torch.minimum(lt,mt)
    lt = torch.norm(cropped_rendered_img-cropped_target_img[:,:,2:(w),0:(h-2)],2,1)
    mt =torch.norm(cropped_rendered_img-cropped_target_img[:,:,0:(w-2),1:(h-1)],2,1)
    b=torch.minimum(lt,mt)
    lt =torch.norm(cropped_rendered_img-cropped_target_img[:,:,1:(w-1),1:(h-1)],2,1)
    mt =torch.norm(cropped_rendered_img-cropped_target_img[:,:,2:(w),1:(h-1)],2,1)
    c = torch.minimum(lt,mt)
    lt =torch.norm(cropped_rendered_img-cropped_target_img[:,:,0:(w-2),2:(h)],2,1)
    mt = torch.norm(cropped_rendered_img-cropped_target_img[:,:,1:(w-1),2:(h)],2,1)
    d = torch.minimum(lt,mt)
    rb = torch.norm(cropped_rendered_img-cropped_target_img[:,:,2:(w),2:(h)],2,1)
   
    a=torch.minimum(a,b)
    b=torch.minimum(c,d)
    a=torch.minimum(a,b)
    a=torch.minimum(a,rb)
    loss_neighbor = torch.mean(a)
    return loss_neighbor
