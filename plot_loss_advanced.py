#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 12 10:34:11 2021

Show train_loss

@author: root
"""

import matplotlib.pyplot as plt
import csv
import argparse
import glob
import os

#hyper-parameters



def open_log(log_filepath):
    try:
        fid_train = open(log_filepath, 'r')
        dir_temp,name = os.path.split(log_filepath)
        fid_test = open(dir_temp+'/'+name.replace('train','test'), 'r')
        
    except:
        print('At least one of the files does not exists, please check your directory.')
    return fid_test, fid_train



def get_info_from_line(line, loss_ith=2):
    if len(line)<=1:
        return False, False
    line=line.replace('p,e,r,:,','')
    
    try:
        line_temp = line.replace(',','').split(' ')
        
        if loss_ith >= len(line_temp):
            return False, False
        if 't,i,m,e'  in line:
            if loss_ith >= len(line_temp)-2:
                return False, False
        count_temp = int(line_temp[1].split(':')[1]  )
        loss_temp = float(line_temp[loss_ith].replace('\n','').replace('"',''))
    except:
        try:
            line_temp  = line.split(',')
            count_temp = int(line_temp[0])
            loss_temp = float(line_temp[loss_ith])
        except:
            '''try:
                line_temp  = line.replace(',','').split(' ')
                count_temp = int(line_temp[2].split(':')[1]  )
                loss_temp = float(line_temp[3])
            except:'''
            print(line)          

    return count_temp, loss_temp

def get_info_from_file(fid,pipeline,end_iter, loss_ith_f):
    mofa_count=[]
    mofa_loss = []
    if pipeline=='EM':
        unet_count=[]
        unet_loss=[]
    begin = 1
    for lines in fid:
        if begin:
            begin = 0
            continue
        try:
            count_temp, loss_temp = get_info_from_line(lines, loss_ith=loss_ith_f)
            if loss_temp ==0:continue
        except:
            pass
        if count_temp >= end_iter or (not count_temp):
            break
        if (('m,o,f,a' in lines) and pipeline=='EM') or pipeline == False:
            mofa_count.append(count_temp)
            mofa_loss.append(loss_temp)
        elif (('u,n,e,t' in lines) and pipeline=='EM'):
            unet_count.append(count_temp)
            unet_loss.append(loss_temp)
    if pipeline=='EM':
        return mofa_count,mofa_loss,unet_count,unet_loss
    else:
        return mofa_count, mofa_loss
            

def draw_log(log_dir,mode = False,ith_loss_f=2):
    dir_search = glob.glob(log_dir +'*_train.csv')
    latest_file = sorted(dir_search, key=os.path.getctime, reverse = True)
    

    end_iter = 999000
    save_dir_name,_ = os.path.split(latest_file[0])

    

    start_iter_train = 999999
    start_iter_test = 999999

    
    mofa_train_count = []
    mofa_train_loss = []
    unet_train_count = []
    unet_train_loss = []
        
    test_loss = []
    test_count = []     
    for list_temp in latest_file:
        fid_test, fid_train = open_log(list_temp)
        mofa_count,mofa_loss = get_info_from_file(fid_test,end_iter=999999999, loss_ith_f=ith_loss_f,pipeline=False)
        test_loss = mofa_loss + test_loss
        test_count = mofa_count +test_count
        
            
        infos = get_info_from_file(fid_train,end_iter=999999999,pipeline=mode, loss_ith_f=ith_loss_f)
        mofa_train_count = infos[0] +mofa_train_count
        mofa_train_loss = infos[1] +mofa_train_loss
        if mode == 'EM':
            unet_train_count = infos[2] +  unet_train_count
            unet_train_loss = infos[3] +unet_train_loss
    return mofa_train_count,mofa_train_loss,unet_train_count,unet_train_loss,test_loss,test_count



par = argparse.ArgumentParser(description='MoFA')

root = '/home/li0005/Program/mofa_unet_newcrop/MoFA_UNet_Save/'
subdirs = [f.path for f in os.scandir(root) if f.is_dir()]
par.add_argument('--log_dir',default="/home/li0005/Program/mofa_unet_newcrop/MoFA_UNet_Save/",type=str,help='The test loss log file')
args = par.parse_args()
for dir_temp in subdirs:
    if 'cut' in os.path.split(dir_temp)[-1]:
        a=1
    if 'UNet' in os.path.split(dir_temp)[-1] or 'basic' in os.path.split(dir_temp)[-1]:
        #continue
        mode_read = 'EM'
        loss_names=['','','overall','landmark','masked_pixel','regularization','area','Perceptual',\
            'Expand_Perceptual','Shrink_Perceptual','Neighbour','Binary']
    else:
        mode_read = False
        loss_names=['','','overall','landmark','masked_pixel','regularization','Perceptual']
    fig, axs = plt.subplots(len(loss_names)-2, 1,figsize=(10,12))
    method_name=dir_temp.split('/')[-1]
    for i_loss in range(2,len(loss_names)+1):
        #plt.subplot(len(loss_names), 1, i_loss+1)
        
        try:
            mofa_train_count,mofa_train_loss,unet_train_count,unet_train_loss,test_loss,test_count=draw_log(dir_temp + '/',mode_read,ith_loss_f=i_loss)
            axs[i_loss-2].plot(mofa_train_count, mofa_train_loss,'r')
            axs[i_loss-2].set_ylabel(loss_names[i_loss])
        
            if len(test_loss)>0:
                axs[i_loss-2].plot(test_count,test_loss,'b')
            if i_loss==2:
                axs[i_loss-2].set_title( method_name)
        except:
            pass
    
