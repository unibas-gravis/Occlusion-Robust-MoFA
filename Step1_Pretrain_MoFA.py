#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 14 15:17:05 2021
Full-face version with 68 3D landmarks
WITH Perceptual Loss
@author: root
"""
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
import os
import argparse
from datetime import date
from facenet_pytorch import InceptionResnetV1
from models import networks

print(networks.__file__)

par = argparse.ArgumentParser(description='Pretrain MoFA')
par.add_argument('--learning_rate',default=0.1,type=float,help='The learning rate')
par.add_argument('--epochs',default=100,type=int,help='Total epochs')
par.add_argument('--batch_size',default=12,type=int,help='Batch sizes')
par.add_argument('--gpu',default=0,type=int,help='The GPU ID')
par.add_argument('--pretrained_model_train',default=000,type=int,help='Pretrained model')
par.add_argument('--img_path',type=str,help='Root of the training samples')

args = par.parse_args()
GPU_no = args.gpu
begin_learning_rate = args.learning_rate


ct = args.pretrained_model_train #load trained model
ct_begin = ct
output_name = 'pretrain_mofa'
device = torch.device("cuda:{}".format(util.device_ids[GPU_no]) if torch.cuda.is_available() else "cpu")

#Hyper parameters
batch = args.batch_size
width = 224
height = 224

epoch = args.epochs
test_batch_num = 3
decay_step_size=5000
decay_rate_gamma =0.99



'''------------------------------------
  Prepare Log Files & Load Models
------------------------------------'''

#prepare log file
today = date.today()
current_path = os.getcwd()  

image_path = (args.img_path + '/' ).replace('//','/')
output_path = current_path+'/MoFA_UNet_Save/'+output_name + '/'
if not os.path.exists(output_path):
    os.makedirs(output_path)

        
loss_log_path_train = output_path+today.strftime("%b-%d-%Y")+"loss_train.csv"
loss_log_path_test = output_path+today.strftime("%b-%d-%Y")+"loss_test.csv"
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



# 3dmm data
model_path = current_path+'/basel_3DMM/model2017-1_bfm_nomouth.h5'
obj = lob.Object3DMM(model_path,device,is_crop = True)
A =  torch.Tensor([[9.06*224/2, 0,  (width-1)/2.0, 0, 9.06*224/2, (height-1)/2.0, 0, 0, 1]]).view(-1, 3, 3).to(device) #intrinsic camera mat
T_ini = torch.Tensor([0, 0, 1000]).to(device)   #camera translation(direction of conversion will be set by flg later)
sh_ini = torch.zeros(3, 9,device=device)    #offset of spherical harmonics coefficient
sh_ini[:, 0] = 0.7 * 2 * math.pi
sh_ini = sh_ini.reshape(-1)


'''--------------------------
  Load Dataset & Networks
--------------------------'''

trainset = load_dataset.CelebDataset(device,image_path, True, height,width,1)
trainset.shuffle()
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch,shuffle=True, num_workers=0)

testset = load_dataset.CelebDataset(device,image_path, False, height,width,1)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch,shuffle=False, num_workers=0)

#renderer and encoder
render_net = ren.Renderer(32)   #block_size^2 pixels are simultaneously processed in renderer, lager block_size consumes lager memory
enc_net = enc.FaceEncoder(obj).to(device)
cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)

# Load ArcFace for perceptual loss
net_recog = networks.define_net_recog(net_recog='r50', pretrained_path='models/ms1mv3_arcface_r50_fp16/backbone.pth')
net_recog = net_recog.to(device)
assert net_recog.training == False
'''----------------------------------
 Fixed Testing Images for Observation
----------------------------------'''
test_input_images = []
test_landmarks = []
for i_test, data_test in enumerate(testloader, 0):
	if i_test >= test_batch_num:
		break
	images, landmarks = data_test
	test_input_images +=[images]
	test_landmarks +=[landmarks]
util.write_tiled_image(torch.cat(test_input_images,dim=0), output_path + 'test_gt.png',10)


# Occlusion Robust Loss for unsupervised initialization
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

def proc(images, landmarks, render_mode):
	'''
	images: network_input
    landmarks: landmark ground truth
    render_mode: renderer mode
	'''
	shape_param, exp_param, color_param, camera_param, sh_param = enc_net(images)
	color_param *= 3    #adjust learning rate
	camera_param[:,:3] *= 0.3
	camera_param[:,5] *= 0.005
	shape_param[:,80:] *= 0 #ignore high dimensional component of BFM
	exp_param[:,64:] *= 0
	color_param[:,80:] *= 0
    
	vertex, color, R, T, sh_coef = enc.convert_params(shape_param, exp_param, color_param, camera_param, sh_param,obj,T_ini,sh_ini,False)
	projected_vertex, sampled_color, shaded_color, occlusion, raster_image, raster_mask = render_net(obj.face, vertex,color, sh_coef, A, R, T, images,ren.RASTERIZE_DIFFERENTIABLE_IMAGE,False, 5, True)
	
	lm68 = projected_vertex[:,0:2,obj.landmark]
	#util.show_tensor_images(raster_image,lm= lm68,batch=batch)
	#util.show_tensor_images(images,lm= landmarks,batch=batch)
	rec_loss, occlusion_fg_mask = occlusionPhotometricLossWithoutBackground(images, raster_image, raster_mask)

	pred_feat = net_recog(image=raster_image,pred_lm=lm68.transpose(1,2))
	gt_feat = net_recog(images,landmarks.transpose(1,2))
	cosine_d = torch.sum(pred_feat * gt_feat, dim=-1)
	perceptual_loss =  torch.sum(1 - cosine_d) / cosine_d.shape[0]
    
	land_loss = torch.mean((obj.weight_lm*(landmarks-lm68))**2)
	
	stat_reg = (torch.sum(shape_param ** 2) + torch.sum(exp_param ** 2) + torch.sum(color_param ** 2))/float(batch)/224.0

	loss=rec_loss*0.5 + 1e-1 * stat_reg + 5e-4 * land_loss + perceptual_loss *0.25

	losses_return = torch.FloatTensor([loss.item(), land_loss.item(), rec_loss.item(), stat_reg.item(),perceptual_loss.item()])
	return loss,losses_return, raster_image,raster_mask,occlusion_fg_mask
#################################################################


'''-----------------------------------------
load pretrained model and continue training
-----------------------------------------'''


if ct!=0:
    
	trained_model_path = output_path+'enc_net_{:06d}.model'.format(ct)
	enc_net = torch.load(trained_model_path, map_location='cuda:{}'.format(util.device_ids[GPU_no]))
	print('Loading pre-trained model:'+ output_path + ' enc_net_{:06d}.model'.format(ct))
	learning_rate_begin=begin_learning_rate*(decay_rate_gamma ** (ct//decay_step_size))
else:
	learning_rate_begin=begin_learning_rate

'''----------
Set Optimizer
----------'''
optimizer = optim.Adadelta(enc_net.parameters(), lr=learning_rate_begin)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer,step_size=decay_step_size,gamma=decay_rate_gamma)
print('Training ...')
start = time.time()
mean_losses = torch.zeros([5])
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
for ep in range(0,epoch):
	for i, data in enumerate(trainloader, 0):


		if (ct-ct_begin) % 500 == 0 :
			'''-------------------------
        	Save Model every 5000 iters
        	--------------------------'''
			if (ct-ct_begin) % 5000 ==0 and ct>ct_begin:
				enc_net.eval()
				torch.save(enc_net, output_path + 'enc_net_{:06d}.model'.format(ct))
        	
			'''-------------------------
        	Save images for observation
        	--------------------------'''
			test_raster_images = []
			test_fg_masks = []
			enc_net.eval()
			with torch.no_grad():
				
				for images, landmarks in zip(test_input_images,test_landmarks):#, test_valid_masks):
				
					l_loss_,_, raster_image, raster_mask, fg_mask = proc(images,landmarks,True)
					
					test_raster_images += [images*(1-raster_mask.unsqueeze(1))+raster_image*raster_mask.unsqueeze(1)]
					test_fg_masks += [fg_mask.reshape(batch,1,224,224)]
				util.write_tiled_image(torch.cat(test_raster_images,dim=0),output_path+'test_image_{}.png'.format(ct),10)
				util.write_tiled_image(torch.cat(test_fg_masks, dim=0), output_path + 'test_image_fgmask_{}.png'.format(ct),10)

            #validating
			'''-------------------------
        	Vlidate Model every 1000 iters
        	--------------------------'''
			if (ct-ct_begin) % 5000 == 0 and ct>ct_begin:
				print('Training mode:'+output_name)
				c_test=0
				mean_test_losses = torch.zeros([5])
				enc_net.eval()
				for i_test, data_test in enumerate(testloader,0):
					
					
					image, landmark = data_test                         
					c_test+=1
					with torch.no_grad():           
						loss_,losses_return_, raster_image, raster_mask, fg_mask = proc(image,landmark,True)
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
		enc_net.train()
		optimizer.zero_grad()
		
		images, landmarks = data
		
		loss,losses_return_, raster_image, raster_mask, fg_mask = proc(images,landmarks,False)
		if images.shape[0]!=batch:
			continue  
		mean_losses += losses_return_
		loss.backward()
		optimizer.step()

		'''-------------------------
        Show Training Loss
        --------------------------'''

		if (ct-ct_begin) % 100 == 0 and (ct-ct_begin)>0:
			end = time.time()
			mean_losses = mean_losses/100
			str = 'train loss:{}'.format(ct)
			for loss_temp in mean_losses:
				str+=' {:05f}'.format(loss_temp)
			str += ' time: {:01f}'.format(end-start)
			print(str)
			writer_train.writerow(str)
			start = end
			mean_losses = torch.zeros([5])
		scheduler.step()
		ct += 1

torch.save(enc_net, output_path + 'enc_net_{:06d}.model'.format(ct))
