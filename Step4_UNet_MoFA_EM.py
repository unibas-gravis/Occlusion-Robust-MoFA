#!/usr/bin/env python3
# -*- coding: utf-8 -*- 
"""
Created on Tue Feb 23 13:12:25 2021

UNet + MoFA
@author: li0005
"""
import os
import torch
import math
import torch.optim as optim
import util.util as util
import csv
import util.load_dataset as load_dataset
import util.load_object as lob
import renderer.rendering as ren
import encoder.encoder as enc
import time
import UNet.UNet as unet
import argparse
from datetime import date
from util.inception_resnet_v1 import InceptionResnetV1
import util.advanced_losses as adlosses

#hyper-parameters
par = argparse.ArgumentParser(description='MoFA')

par.add_argument('--learning_rate',default=0.06,type=float,help='The learning rate')
par.add_argument('--epochs',default=130,type=int,help='Total epochs')
par.add_argument('--batch_size',default=6,type=int,help='Batch sizes')
par.add_argument('--gpu',default=0,type=int,help='The GPU ID')
par.add_argument('--pretrained_model',default=0,type=int,help='Pretrained model')
par.add_argument('--img_path',type=str,help='Root of the training samples')



args = par.parse_args()
GPU_no = args.gpu
ct = args.pretrained_model #load trained mofa model
output_name = 'MoFA_UNet'
device = torch.device("cuda:{}".format(util.device_ids[GPU_no]) if torch.cuda.is_available() else "cpu")


begin_learning_rate = args.learning_rate
decay_step_size=5000
decay_rate_gamma =0.99
decay_rate_unet =0.95
learning_rate_begin=begin_learning_rate*(decay_rate_gamma ** ((300000)//decay_step_size)) 
mofa_lr_begin = learning_rate_begin * (decay_rate_gamma ** (ct//decay_step_size))
unet_lr_begin = learning_rate_begin *0.06* (decay_rate_unet**(ct//decay_step_size))

ct_begin=ct

today = date.today()
current_path = os.getcwd()  
model_path = current_path+'/basel_3DMM/model2017-1_bfm_nomouth.h5'

image_path = args.img_path
output_path = current_path+'/MoFA_UNet_Save/'+output_name + '/'
if not os.path.exists(output_path):
    os.makedirs(output_path)
    

loss_log_path_train = output_path+today.strftime("%b-%d-%Y")+"loss_train.csv"
loss_log_path_test = output_path+today.strftime("%b-%d-%Y")+"loss_test.csv"

'''------------------
  Prepare Log Files
------------------'''
if ct != 0:
    try:
        fid_train = open(loss_log_path_train, 'a')
        fid_test = open(loss_log_path_test, 'a')
    except:
    	fid_train = open(loss_log_path_train, 'w')
    	fid_test = open(loss_log_path_test, 'w')
else:
    fid_train = open(loss_log_path_train, 'w')
    fid_test = open(loss_log_path_test, 'w')
writer_train = csv.writer(fid_train, lineterminator="\r\n")
writer_test = csv.writer(fid_test, lineterminator="\r\n")

'''------------------
  Load Data & Models
------------------'''
#parameters
batch = args.batch_size
width = 224
height = 224

epoch = args.epochs
test_batch_num = 10


A =  torch.Tensor([[250, 0, (width-1)/2.0, 0, 250, (height-1)/2.0, 0, 0, 1]]).view(-1, 3, 3).to(device) #intrinsic camera mat

T_ini = torch.Tensor([0, 30, 300]).to(device)   #camera translation(direction of conversion will be set by flg later)
sh_ini = torch.zeros(3, 9,device=device)    #offset of spherical harmonics coefficient
sh_ini[:, 0] = 0.7 * 2 * math.pi
sh_ini = sh_ini.reshape(-1)


cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)


'''-------------
  Load Dataset
-------------'''

testset = load_dataset.CelebDataset(device,image_path, False, height,width,1)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch,shuffle=False, num_workers=0)

# 3dmm data
obj = lob.Object3DMM(model_path, device)


'''----------------
Prepare Network and Optimizer
----------------'''

#renderer and encoder and UNet
render_net = ren.Renderer(32)   #block_size^2 pixels are simultaneously processed in renderer, lager block_size consumes lager memory
enc_net = enc.FaceEncoder(obj).to(device)
unet_for_mask = unet.UNet(in_ch=6, out_ch=1).to(device)
resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)
'''----------------------------------
 Fixed Testing Images for Observation
----------------------------------'''
test_input_images = []
test_landmarks = []
test_landmark_masks = []
for i_test, data_test in enumerate(testloader, 0):
	if i_test >= test_batch_num:
		break
	images, landmarks = data_test
	test_input_images +=[images]
	test_landmarks +=[landmarks]
util.write_tiled_image(torch.cat(test_input_images,dim=0), output_path + 'test_gt.png',10)


def occlusionPhotometricLossWithoutBackground(gt,rendered,fgmask,standardDeviation=0.043,backgroundStDevsFromMean=3.0):
	normalizer = (-3 / 2 * math.log(2 * math.pi) - 3 * math.log(standardDeviation))
	fullForegroundLogLikelihood = (torch.sum(torch.pow(gt - rendered,2), axis=1)) * -0.5 / standardDeviation / standardDeviation + normalizer
	uniformBackgroundLogLikelihood = math.pow(backgroundStDevsFromMean * standardDeviation, 2) * -0.5 / standardDeviation / standardDeviation + normalizer
	occlusionForegroundMask = fgmask * (fullForegroundLogLikelihood > uniformBackgroundLogLikelihood).type(torch.FloatTensor).cuda(util.device_ids[GPU_no])
	foregroundLogLikelihood = occlusionForegroundMask*fullForegroundLogLikelihood
	lh = torch.mean(foregroundLogLikelihood)
	return -lh, occlusionForegroundMask


'''-------------
Network Forward
-------------'''
#################################################################
def proc_mofaunet(images, landmarks, render_mode,train_net=False):

	shape_param, exp_param, color_param, camera_param, sh_param = enc_net(images)
	color_param *= 3    #adjust learning rate
	camera_param[:,:3] *= 0.3
	camera_param[:,5] *= 0.005
	shape_param[:,80:] *= 0 #ignore high dimensional component of BFM
	exp_param[:,64:] *= 0
	color_param[:,80:] *= 0
    
	vertex, color, R, T, sh_coef = enc.convert_params(shape_param, exp_param, color_param, camera_param, sh_param,obj,T_ini,sh_ini,False)
	projected_vertex, sampled_color, shaded_color, occlusion, raster_image, raster_mask = render_net(obj.face, vertex,color, sh_coef, A,R, T, images,ren.RASTERIZE_DIFFERENTIABLE_IMAGE,False, 5, True)
	#render vertex color and sample corresponding pixel
	'''-------------------------------------
	U-Net input: Raster [RGB] + ORG [RGB]
	----------------------------------------'''

	image_concatenated = torch.cat(( raster_image,images),axis = 1)
	unet_est_mask = unet_for_mask(image_concatenated)
	valid_loss_mask = raster_mask.unsqueeze(1)*unet_est_mask

	
	masked_rec_loss = torch.mean(torch.sum(torch.norm(valid_loss_mask*(images - raster_image), 2, 1)) / torch.clamp(torch.sum(raster_mask.unsqueeze(1)*unet_est_mask),min=1))
	bg_unet_loss = torch.mean(torch.sum(raster_mask.unsqueeze(1)*(1-unet_est_mask),axis=[2,3])/torch.clamp(torch.sum(raster_mask.unsqueeze(1),axis=[2,3]),min=1))

	mask_binary_loss= torch.zeros([1])
	perceptual_loss= torch.zeros([1])
	land_loss= torch.zeros([1])
	stat_reg= torch.zeros([1])
	if train_net=='unet':
		mask_binary_loss= 10*(0.5-1 * torch.mean(torch.norm(valid_loss_mask-0.5,2,1)))
		loss_unet = mask_binary_loss + ( 10**-1* bg_unet_loss*2.5 )*2
	if train_net == 'mofa':
		id_gt = resnet(images*  raster_mask.unsqueeze(1))
		id_reconstruction = resnet(raster_image)
		perceptual_loss = torch.mean(1-cos(id_reconstruction, id_gt))
		land_loss = torch.mean((obj.weight_lm*(landmarks-projected_vertex[:,0:2,obj.landmark]))**2)
		stat_reg = (torch.sum(shape_param ** 2) + torch.sum(exp_param ** 2) + torch.sum(color_param ** 2))/float(batch)/224.0
		loss_mofa = masked_rec_loss*4 + perceptual_loss/2+ 10 ** -2 * stat_reg/1.5 + 10 ** -2 * land_loss +( 10**-1* bg_unet_loss*2.5 )*2

    

	if train_net == False:
		mask_binary_loss= 5*(0.5- torch.mean(torch.norm(valid_loss_mask-0.5,2,1)))#torch.minimum(torch.norm(valid_loss_mask-1,2), torch.norm(valid_loss_mask-1,2))
		id_gt = resnet(images*  raster_mask.unsqueeze(1))
		id_reconstruction = resnet(raster_image)
		perceptual_loss = torch.mean(1-cos(id_reconstruction, id_gt))
		land_loss = torch.mean((obj.weight_lm*(landmarks-projected_vertex[:,0:2,obj.landmark]))**2)
		stat_reg = (torch.sum(shape_param ** 2) + torch.sum(exp_param ** 2) + torch.sum(color_param ** 2))/float(batch)/224.0

		loss_test = mask_binary_loss + (masked_rec_loss*2 + 10**-1* bg_unet_loss*2.5 )*2 + perceptual_loss/2+ 10 ** -2 * stat_reg/1.5 + 10 ** -2 * land_loss 


	I_IM_Per_loss = torch.zeros([1])
	IRM_IM_Per_loss = torch.zeros([1])
	if train_net=='unet' or train_net == False:
		I_target_masked=images*  valid_loss_mask
		id_target_masked = resnet(I_target_masked, is_shallow=True )
		id_target = resnet(images, is_shallow=True )
		id_reconstruct_masked = resnet(raster_image*  valid_loss_mask , is_shallow=True )
		I_IM_Per_loss = torch.mean(1-cos(id_target, id_target_masked))
		IRM_IM_Per_loss = torch.mean(1-cos(id_reconstruct_masked, id_target_masked))
		if train_net=='unet':
			loss_unet += I_IM_Per_loss*2.5 + IRM_IM_Per_loss*3
		if train_net == False:
			loss_test+= I_IM_Per_loss*2.5 + IRM_IM_Per_loss*3

	
    #force it to be binary mask
	loss_mask_neighbor = torch.zeros([1])
	if train_net=='unet':
		loss_mask_neighbor = adlosses.neighbor_unet_loss(images, valid_loss_mask, raster_image)
		loss_unet +=loss_mask_neighbor*15
	if train_net=='unet':
		loss=loss_unet
	if train_net == 'mofa':
		loss=loss_mofa
	if train_net == False:
		loss=loss_test
	losses_return = torch.FloatTensor([loss.item(),land_loss.item(),masked_rec_loss.item(),stat_reg.item(),bg_unet_loss.item(),\
                                       perceptual_loss.item(),I_IM_Per_loss.item(),IRM_IM_Per_loss.item(),loss_mask_neighbor.item(),mask_binary_loss.item()])

	return loss, losses_return, raster_image,raster_mask, valid_loss_mask
#################################################################

'''-----------------------------------------
load pretrained model and continue training
-----------------------------------------'''
if ct!=0: 
    trained_model_path = output_path+'enc_net_{:06d}.model'.format(ct)
    enc_net = torch.load(trained_model_path, map_location='cuda:{}'.format(util.device_ids[GPU_no]))
    unet_model_path = output_path+'unet_{:06d}.model'.format(ct)
    unet_for_mask = torch.load(unet_model_path, map_location='cuda:{}'.format(util.device_ids[GPU_no]))
else:
    trained_model_path = current_path+'/MoFA_UNet_Save/pretrain_mofa/enc_net_300000.model'
    enc_net = torch.load(trained_model_path, map_location='cuda:{}'.format(util.device_ids[GPU_no]))
    unet_model_path = current_path+'/MoFA_UNet_Save/Pretrain_UNet/unet_mask_100000.model'
    unet_for_mask = torch.load(unet_model_path, map_location='cuda:{}'.format(util.device_ids[GPU_no]))
print('Loading pre-trained unet: \n'+trained_model_path +'\n' +unet_model_path)




'''----------
Set Optimizer
----------'''
optimizer_mofa = optim.Adadelta(enc_net.parameters(), lr=mofa_lr_begin)
optimizer_unet = optim.Adadelta(unet_for_mask.parameters(), lr=unet_lr_begin)
scheduler_mofa = torch.optim.lr_scheduler.StepLR(optimizer_mofa,step_size=decay_step_size,gamma=0.99)
scheduler_unet = torch.optim.lr_scheduler.StepLR(optimizer_unet,step_size=decay_step_size,gamma=decay_rate_unet)

print('Training ...')
start = time.time()
mean_losses_mofa = torch.zeros([10])
mean_losses_unet = torch.zeros([10])

trainset = load_dataset.CelebDataset(device,image_path, True, height,width,1)

for ep in range(0,epoch):

	
	trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch,shuffle=True, num_workers=0)
	
	for i, data in enumerate(trainloader, 0):

		'''-------------------------
        Save images for observation
        --------------------------'''
		if (ct-ct_begin) % 500 == 0:
			enc_net.eval()
			unet_for_mask.eval()
			test_raster_images = []
			valid_loss_mask_temp = []

			for images, landmarks in zip(test_input_images,test_landmarks ):

				with torch.no_grad():
					_, _, raster_image,raster_mask, valid_loss_mask = proc_mofaunet(images,landmarks,True,False)
					
					test_raster_images += [images*(1-raster_mask.unsqueeze(1))+raster_image*raster_mask.unsqueeze(1)]
					valid_loss_mask_temp += [valid_loss_mask]
					
			util.write_tiled_image(torch.cat(test_raster_images,dim=0),output_path+'test_image_{}.png'.format(ct),10)
			util.write_tiled_image(torch.cat(valid_loss_mask_temp , dim=0), output_path + 'valid_loss_mask_{}.png'.format(ct),10)

			'''-------------------------
             Save Model every 5000 iters
          	--------------------------'''
			if (ct-ct_begin) % 5000 ==0 and ct>ct_begin:
				torch.save(enc_net, output_path + 'enc_net_{:06d}.model'.format(ct))
				torch.save(unet_for_mask, output_path + 'unet_{:06d}.model'.format(ct))
            
            
            #validating
			'''-------------------------
        	Validate Model every 1000 iters
        	--------------------------'''
			if (ct-ct_begin) % 1000 == 0 and ct>ct_begin:
				print('Training mode:'+output_name)
				c_test=0
				mean_test_losses = torch.zeros([10])
				
				for i_test, data_test in enumerate(testloader,0):
    
					image, landmark = data_test
					c_test+=1
					with torch.no_grad():
						loss_, losses_return_,_,_,_ = proc_mofaunet(image,landmark,True,False)
						mean_test_losses += losses_return_
				mean_test_losses = mean_test_losses/c_test
				str = 'test loss:{}'.format(ct)
				for loss_temp in losses_return_:
					str+=' {:05f}'.format(loss_temp)
				print(str)
				writer_test.writerow(str)
	
			fid_train.close()
			fid_train = open(loss_log_path_train , 'a')
			writer_train = csv.writer(fid_train, lineterminator="\r\n")

			fid_test.close()
			fid_test = open(loss_log_path_test, 'a')
			writer_test = csv.writer(fid_test, lineterminator="\r\n")

		'''-------------------------
        Model Training
        --------------------------'''
		
		
		images, landmarks = data
		if images.shape[0]!=batch:
			continue 
		if ct %30000 >5000:
			enc_net.train()
			unet_for_mask.eval()
			loss_mofa, losses_return_mofa, _,_,_= proc_mofaunet(images,landmarks,True,'mofa')
			loss_mofa.backward()
			optimizer_mofa.step()
			mean_losses_mofa+= losses_return_mofa
		else:
			unet_for_mask.train()
			enc_net.eval()
            
			loss_unet, losses_return_unet, _,_,_= proc_mofaunet(images,landmarks,True,'unet')
			loss_unet.backward()
			optimizer_unet.step()
			mean_losses_unet+= losses_return_unet
		ct += 1
		scheduler_unet.step()
		scheduler_mofa.step()
		optimizer_unet.zero_grad()
		optimizer_mofa.zero_grad()

		
		
		'''-------------------------
        Show Training Loss
        --------------------------'''

		if (ct-ct_begin) % 100 == 0 and ct>ct_begin:
			end = time.time()
			mean_losses_unet = mean_losses_unet/100
			mean_losses_mofa = mean_losses_mofa/100
			str = 'mofa loss:{}'.format(ct)
			for loss_temp in mean_losses_mofa:
				str+=' {:05f}'.format(loss_temp)
			str += '\nunet loss:{}'.format(ct)
			for loss_temp in mean_losses_unet:
				str+=' {:05f}'.format(loss_temp)
			str += ' time: {}'.format(end-start)
			print(str)
			writer_train.writerow(str)
			start = time.time()
			mean_losses_unet = torch.zeros([10])
			mean_losses_mofa = torch.zeros([10])


torch.save(enc_net, output_path + 'enc_net_{:06d}.model'.format(ct))
