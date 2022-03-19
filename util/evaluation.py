#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 22 22:10:30 2021

---------------------------
Functions for Evaluation

---------------------------
@author: li0005
"""
import torch
def get_mask_area(mask):
    #mask: ? x C x W x H
    #output: area of face region, N x 1
    area = torch.sum(mask,axis=[2,3])
    area = torch.mean(area,dim=1)
    area = torch.clamp(area,min=1)
    return area

def mask_accuracy(GT_mask,est_mask,rendered_mask):
    TP = GT_mask*est_mask*rendered_mask
    TN = (1-GT_mask) * (1-est_mask)*rendered_mask
    FN = GT_mask * (1-est_mask)*rendered_mask
    FP = (1-GT_mask) * est_mask*rendered_mask
    _,_,w,h = GT_mask.size()
    area_TP = get_mask_area(TP)
    area_TN = get_mask_area(TN)
    area_FN = get_mask_area(FN)
    area_FP = get_mask_area(FP)
    accuracy = torch.mean(torch.div((area_TP + area_TN) , (area_TP+area_TN +area_FP+area_FN)))
    
    precision = torch.mean(torch.div(area_TP , (area_TP + area_FP) )) # Accuracy of predicted face region
    F1_score = torch.mean(torch.div(2 * area_TP , (2 * area_TP +area_FP+area_FN)))
    recall = torch.mean(torch.div(area_TP , (area_TP + area_FN) ))
    return accuracy, precision, F1_score, recall
    
    

def pixel_valid_MAE(img_org,img_reconstruted, face_mask):
    abs_err = torch.abs(face_mask*(img_org - img_reconstruted) )
    rec_loss = 255*torch.mean( torch.sum(abs_err,axis = [1,2,3])/ torch.clamp(torch.sum(face_mask,axis=[1,2,3]),min=1))
    _,c,_,_ = face_mask.shape
    if c == 1:
       rec_loss = rec_loss/3 
    return rec_loss
