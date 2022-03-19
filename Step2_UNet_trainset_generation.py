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
par = argparse.ArgumentParser(description='Generate training set for UNet')
par.add_argument('--pretrained_MoFA',default = './MoFA_UNet_Save/pretrain_mofa/enc_net_300000.model',type=str,help='Path of the pre-trained model')
par.add_argument('--gpu',default=0,type=int,help='The GPU ID')
par.add_argument('--img_path',type=str,help='Root of the training samples')

args = par.parse_args()
GPU_no = args.gpu
trained_model_path = args.pretrained_MoFA
output_name = 'UNet_trainset'

image_path = (args.img_path + '/' ).replace('//','/')
current_path = os.getcwd()  
model_path = current_path+'/basel_3DMM/model2017-1_bfm_nomouth.h5'
save_path = current_path+'/MoFA_UNet_Save/'+output_name + '/'
if not os.path.exists(save_path):
    os.makedirs(save_path)
    
device = torch.device("cuda:{}".format(util.device_ids[GPU_no ]) if torch.cuda.is_available() else "cpu")




#parameters
batch = 32
width = height = 224


A =  torch.Tensor([[9.06*224/2, 0,  (width-1)/2.0, 0, 9.06*224/2, (height-1)/2.0, 0, 0, 1]]).view(-1, 3, 3).to(device) #intrinsic camera mat
T_ini = torch.Tensor([0, 0, 1000]).to(device)   #camera translation(direction of conversion will be set by flg later)
sh_ini = torch.zeros(3, 9,device=device)    #offset of spherical harmonics coefficient
sh_ini[:, 0] = 0.7 * 2 * math.pi
sh_ini = sh_ini.reshape(-1)

# 3dmm data
obj = lob.Object3DMM(model_path, device=device,is_crop = True)

#renderer and encoder
render_net = ren.Renderer(32)   #block_size^2 pixels are simultaneously processed in renderer, lager block_size consumes lager memory
enc_net = enc.FaceEncoder(obj).to(device)


def occlusionPhotometricLossWithoutBackground(gt,rendered,fgmask,standardDeviation=0.043,backgroundStDevsFromMean=3.0):
	normalizer = (-3 / 2 * math.log(2 * math.pi) - 3 * math.log(standardDeviation))
	fullForegroundLogLikelihood = (torch.sum(torch.pow(gt - rendered,2), axis=1)) * -0.5 / standardDeviation / standardDeviation + normalizer
	uniformBackgroundLogLikelihood = math.pow(backgroundStDevsFromMean * standardDeviation, 2) * -0.5 / standardDeviation / standardDeviation + normalizer
	occlusionForegroundMask = fgmask * (fullForegroundLogLikelihood > uniformBackgroundLogLikelihood).type(torch.FloatTensor).cuda(util.device_ids[GPU_no ])
	foregroundLogLikelihood = occlusionForegroundMask*fullForegroundLogLikelihood
	lh = torch.mean(foregroundLogLikelihood)
	return -lh, occlusionForegroundMask



#################################################################
def proc(images, render_mode):
	shape_param, exp_param, color_param, camera_param, sh_param = enc_net(images)
	color_param *= 3    #adjust learning rate
	camera_param[:,:3] *= 0.3
	camera_param[:,5] *= 0.005
	shape_param[:,80:] *= 0 #ignore high dimensional component of BFM
	exp_param[:,64:] *= 0
	color_param[:,80:] *= 0
	vertex, color, R, T, sh_coef = enc.convert_params(shape_param, exp_param, color_param, camera_param, sh_param,obj,T_ini,sh_ini,False)
	projected_vertex, sampled_color, shaded_color, occlusion, raster_image, raster_mask = render_net(obj.face, vertex,color, sh_coef, A, R, T, images,ren.RASTERIZE_DIFFERENTIABLE_IMAGE,False, 5, True)
		
	_, occlusion_fg_mask = occlusionPhotometricLossWithoutBackground(images, raster_image, raster_mask)

	return raster_image,occlusion_fg_mask
#################################################################


enc_net = torch.load(trained_model_path , map_location='cuda:{}'.format(util.device_ids[GPU_no ]))

        
landmark_file = image_path + '../all_landmarks.csv'
testset = load_dataset.CelebDataset(device,image_path, False, height,width,1,landmark_file,test_mode=True)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch,shuffle=False, num_workers=0)


for i, data in enumerate(testloader, 0):
        
    with torch.no_grad():
    	images,image_paths,_,lms = data
    	#util.show_tensor_images(images,lm= lms,batch=2)
    	enc_net.eval()
    	image_raster,occlusion_fg_mask = proc(images,True)
    	
    	img_num,ch,height,width = image_raster.shape
        
    	for iter_save in range(img_num):

        	_,filename_save=os.path.split(image_paths[iter_save])
                
        	img_result = image_raster[iter_save].detach().to('cpu').numpy()
        	img_result = np.flip(np.swapaxes(np.swapaxes(img_result, 0, 1), 1, 2) * 255, 2)
        	cv2.imwrite(save_path+filename_save.replace('.jpg','_raster.jpg'),img_result)
            
        	img_org = images[iter_save].detach().to('cpu').numpy()
        	img_org = np.flip(np.swapaxes(np.swapaxes(img_org, 0, 1), 1, 2) * 255, 2)
        	cv2.imwrite(save_path+filename_save.replace('.jpg','_org.jpg'),img_org)
                                                

        	img_mask = occlusion_fg_mask[iter_save].detach().to('cpu').numpy()
        	cv2.imwrite(save_path+filename_save.replace('.jpg','_mask.jpg'),img_mask*255)

