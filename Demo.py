#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep  3 10:42:21 2021

@author: li0005
"""

import torch
import os
import math
import util.util as util
import util.load_dataset as load_dataset
import util.load_object as lob
import renderer.rendering as ren
import encoder.encoder as enc
import cv2
import numpy as np
import argparse
import util.evaluation as evaluation

par = argparse.ArgumentParser(description='Test: MoFA+UNet ')
par.add_argument('--pretrained_model_test',type=str,help='Path of the pre-trained model')
par.add_argument('--gpu',default=0,type=int,help='The GPU ID')
par.add_argument('--test_mode',type=str,help='Model name')
par.add_argument('--test_path',type=str,help='Root of the testing samples')
par.add_argument('--save_path',type=str,help='Root to save samples')
par.add_argument('--landmark_list_name',default='../test_landmarks.csv',type=str,help='Root of the training samples')


args = par.parse_args()
landmark_list_name = args.landmark_list_name
output_path = args.save_path
test_img_dir = (args.test_path + '/' ).replace('//','/')
GPU_no = args.gpu
trained_mofa_path = args.pretrained_model_test
test_mode = args.test_mode
device = torch.device("cuda:{}".format(util.device_ids[GPU_no ]) if torch.cuda.is_available() else "cpu")

#parameters
batch = 6
width = height = 224

current_path = os.getcwd()  
model_path = current_path+'/basel_3DMM/model2017-1_bfm_nomouth.h5'
obj_intact = lob.Object3DMM(model_path,device)
obj_cropped = lob.Object3DMM(model_path,device,is_crop = True)
A =  torch.Tensor([[9.06*224/2, 0,  (width-1)/2.0, 0, 9.06*224/2, (height-1)/2.0, 0, 0, 1]]).view(-1, 3, 3).to(device) #intrinsic camera mat
T_ini = torch.Tensor([0, 0, 1000]).to(device)   #camera translation(direction of conversion will be set by flg later)
sh_ini = torch.zeros(3, 9,device=device)    #offset of spherical harmonics coefficient
sh_ini[:, 0] = 0.7 * 2 * math.pi
sh_ini = sh_ini.reshape(-1)

# 3dmm data
triangles_intact = obj_intact.face.detach().to('cpu').numpy().T
#renderer and encoder
render_net = ren.Renderer(32)   #block_size^2 pixels are simultaneously processed in renderer, lager block_size consumes lager memory
render_net_cropped = ren.Renderer(32) 

def occlusionPhotometricLossWithoutBackground(gt,rendered,fgmask,standardDeviation=0.043,backgroundStDevsFromMean=3.0):
	normalizer = (-3 / 2 * math.log(2 * math.pi) - 3 * math.log(standardDeviation))
	fullForegroundLogLikelihood = (torch.sum(torch.pow(gt - rendered,2), axis=1)) * -0.5 / standardDeviation / standardDeviation + normalizer
	uniformBackgroundLogLikelihood = math.pow(backgroundStDevsFromMean * standardDeviation, 2) * -0.5 / standardDeviation / standardDeviation + normalizer
	occlusionForegroundMask = fgmask * (fullForegroundLogLikelihood > uniformBackgroundLogLikelihood).type(torch.FloatTensor).cuda(util.device_ids[GPU_no ])
	foregroundLogLikelihood = occlusionForegroundMask*fullForegroundLogLikelihood
	lh = torch.mean(foregroundLogLikelihood)
	return -lh, occlusionForegroundMask


#main processing
#################################################################
def proc(images, occlusion_mode=False,landmark_visible_mask=None, valid_mask=None,image_org=None,where_occmask=None):
    #valid_mask: 1 indicating unoccluded part of faces, vice versa
	'''
	images: network_input
    landmarks: landmark ground truth
    render_mode: renderer mode
    occlusion mode: use occlusion robust loss
	'''
	shape_param, exp_param, color_param, camera_param, sh_param = enc_net(images)
	color_param *= 3    #adjust learning rate
	camera_param[:,:3] *= 0.3
	camera_param[:,5] *= 0.005
	shape_param[:,80:] *= 0 #ignore high dimensional component of BFM
	exp_param[:,64:] *= 0
	color_param[:,80:] *= 0
    
	#vertex, color, R, T, sh_coef = enc.convert_params(shape_param, exp_param*0, color_param, camera_param, sh_param,obj,T_ini,sh_ini,False)
	nonexp_vertex_intact, color, R, T, sh_coef = enc.convert_params_noexp(shape_param, exp_param*0, color_param, camera_param, sh_param,obj_intact,T_ini,sh_ini,False)
	
	projected_vertex_intact, _,_, _, raster_image_intact, _ = render_net(obj_intact.face, nonexp_vertex_intact,color, sh_coef, A, R, T, images,ren.RASTERIZE_DIFFERENTIABLE_IMAGE,False, 5, True)
	lmring = nonexp_vertex_intact[:,:,obj_intact.ringnet_lm]
    
	vertex_cropped, color, R, T, sh_coef = enc.convert_params(shape_param, exp_param, color_param, camera_param, sh_param,obj_cropped,T_ini,sh_ini,False)
	_, _, _, _, raster_image_fitted, raster_mask = render_net_cropped(obj_cropped.face, vertex_cropped,color, sh_coef, A, R, T, images,ren.RASTERIZE_DIFFERENTIABLE_IMAGE,False, 5, True)
    
	if where_occmask== 'Occrobust':
		rec_loss, occlusion_fg_mask = occlusionPhotometricLossWithoutBackground(images, raster_image_fitted, raster_mask)
	elif where_occmask== 'UNet':
		image_concatenated = torch.cat(( raster_image_fitted,images),axis = 1)
		unet_est_mask = unet_for_mask(image_concatenated)
		occlusion_fg_mask = raster_mask.unsqueeze(1)*unet_est_mask
	else:
		print('No way to generate occlusion segmentation masks.')
		occlusion_fg_mask=False
	
	#util.show_tensor_images(raster_image_intact,lmring)
	raster_image_fitted = images*(1-raster_mask.unsqueeze(1))+raster_image_fitted*raster_mask.unsqueeze(1)
	return raster_image_fitted,raster_mask ,lmring,nonexp_vertex_intact,occlusion_fg_mask

#################################################################

def write_lm_txt(filename_save,lms_ringnet_temp):
    anno_file = open( filename_save,"w")
    
    lms  = lms_ringnet_temp.T.astype(np.int)
            
    str_temp = ''
    for i_temp in range(7):
        str_temp +='{} {} {}\n'.format(lms[i_temp,0],lms[i_temp,1],lms[i_temp,2])
    anno_file.write(str_temp)    
    anno_file.close()
    

    
def write_txt(filename_save,string):
    anno_file = open( filename_save,"w")
    anno_file.write(string)    
    anno_file.close()
    
    
enc_net = torch.load(trained_mofa_path.replace('/unet_','/enc_net_') , map_location='cuda:{}'.format(util.device_ids[GPU_no ]))
try:
    unet_model_path =trained_mofa_path.replace('enc_net_','unet_')
    unet_for_mask = torch.load(unet_model_path, map_location='cuda:{}'.format(util.device_ids[GPU_no]))
    where_mask='UNet'
except:
    where_mask='Occrobust'



if not os.path.exists(output_path):
    os.mkdir(output_path)


mask_MAE_losses = []
separate_err = ''
image_path = (test_img_dir + '/').replace('//','/')
        
if not os.path.exists(output_path):
    os.mkdir(output_path)
landmark_file = image_path + landmark_list_name
if not os.path.exists(landmark_file):
    landmark_file =  landmark_list_name
if not os.path.exists(landmark_file):
    print("Please enter the path of the landmark list.")
testset = load_dataset.CelebDataset(device,image_path, False, height,width,1,landmark_file,test_mode =True)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch,shuffle=False, num_workers=0)
#-----testing-----
test_raster_images = []
test_fg_masks = []
im_iter=0
    
str_lms = ''
'''---------------------------------------------------
Reconstructed results, masks
---------------------------------------------------'''
with torch.no_grad():
    enc_net.eval()
    for i, data in enumerate(testloader, 0):

        images,image_paths,valid_mask,lm = data
            
        image_results,raster_mask,lmring,vertex_3d,occlusion_fg_mask= proc(images,where_occmask=where_mask)
        
            
        
        img_num,_,_,_ = image_results.shape
        for iter_save in range(img_num):
            filename_save=image_paths[iter_save].replace(test_img_dir,output_path)
            dir_temp,name_org = os.path.split(filename_save)
            if valid_mask[0].dim()==3:
                    
                accuracy, precision, F1_score, recall = evaluation.mask_accuracy(valid_mask[iter_save].unsqueeze(0), occlusion_fg_mask[iter_save].unsqueeze(0), raster_mask[iter_save].unsqueeze(0))
                
                mask_MAE_temp = [accuracy.item(),precision.item(),F1_score.item(), recall.item()]
                mask_MAE_losses.append(mask_MAE_temp)
                separate_err += name_org+ ' {:.5f} {:.5f} {:.5f} {:.5f}\n'.format(mask_MAE_temp[0], mask_MAE_temp[1], mask_MAE_temp[2],mask_MAE_temp[3])
                
            
                
            if not os.path.exists(dir_temp):
                os.makedirs(dir_temp)
            
            img_result = image_results[iter_save].transpose(0,1).transpose(1,2).detach().to('cpu').numpy()* 255
            img_result = np.flip(img_result , 2)
            cv2.imwrite(filename_save.replace('.jpg','_result.jpg'),img_result)

            #Save target images
            img_org = images[iter_save].transpose(0,1).transpose(1,2).detach().to('cpu').numpy()*255
            img_org = np.flip(img_org, 2)
            cv2.imwrite(filename_save.replace('.jpg','_org.jpg'),img_org)
                                                
            # Save segmentation masks
            img_mask =np.round(occlusion_fg_mask[iter_save].detach().to('cpu').numpy())*255
            cv2.imwrite(filename_save.replace('.jpg','_estmask.jpg'),img_mask)
                
            # Save landmarks for NoW evaluation
            lms_ringnet_temp = lmring.detach().to('cpu').numpy()[iter_save]
            write_lm_txt(filename_save.replace('.jpg','.txt'),lms_ringnet_temp)

            # Save vertices for NoW evaluation
            vertexes_temp = vertex_3d[iter_save].detach().to('cpu').numpy()
            util.write_obj_with_colors(filename_save.replace('.jpg','.obj'), vertexes_temp.T, triangles_intact)
            im_iter+=1
            print('No. {}'.format(im_iter))
if valid_mask[0].dim()==3:
    mask_losses = np.mean(np.array(mask_MAE_losses),0)
    mask_std_losses = np.std(np.array(mask_MAE_losses),0)
    mask_err += test_mode +' '+ test_img_dir+' {:.2f} {:.2f} {:.2f} {:.2f} delta: {:.2f} {:.2f} {:.2f} {:.2f}\n'\
            .format(mask_losses[0], mask_losses[1], mask_losses[2],mask_losses[3],\
                    mask_std_losses[0],mask_std_losses[1],mask_std_losses[2],mask_std_losses[3])

mask_file = open(output_path+test_mode + '_mask_evaluation.txt',"w")
mask_file.write(mask_err)
mask_file.close()

separate_file = open(output_path+test_mode + '_Mask_separate.txt',"w")
separate_file.write(separate_err)
separate_file.close()

