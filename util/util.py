import torch
import numpy as np
import cv2 as cv
import sys
# import configparser
import os
import cv2

device_ids=[0,1,2,3]

def get_all_files_endwith(image_dir,ending):
    '''------------------------------------------------------------------------
    get all files end with 'ending' under a directory, 
    including sub, subsub ... directories
    --------------------------------------------------------------------------'''
    paths = []
    for dirpath, dirnames, filenames in os.walk(image_dir):
        for filename in [f for f in filenames if f.endswith(ending)]:
            paths.append(os.path.join(dirpath, filename))
    return paths
    
def write_obj_with_colors(obj_name, vertices, triangles, colors=None):
    ''' Save 3D face model with texture represented by colors.
    Args:
        obj_name: str
        vertices: N x 3
        colors: N x 3
        triangles: N x 3
    '''
    triangles = triangles.copy()
    triangles += 1 # meshlab start with 1
    
    if obj_name.split('.')[-1] != 'obj':
        obj_name = obj_name + '.obj'
        
    # write obj
    with open(obj_name, 'w') as f:
        
        # write vertices & colors
        for i in range(vertices.shape[0]):
            s = 'v {} {} {} \n'.format(vertices[i, 0], vertices[i, 1], vertices[i, 2])
            f.write(s)

        # write f: ver ind/ uv ind
        [k, ntri] = triangles.shape
        for i in range(triangles.shape[0]):
            s = 'f {} {} {}\n'.format(triangles[i, 2], triangles[i, 1], triangles[i, 0])
            f.write(s)

font = cv2.FONT_HERSHEY_SIMPLEX 


def show_normal_images(img,lm= None,batch=12):
    '''
    img: B x 3 x W x H
    lm: B x 2 x 68
    '''

    image_rec=img
    
    for image_iter in range(batch):
        temp_rec = (image_rec.copy()*255).astype(np.uint8)
        if not lm is None:
            lm_temp=np.reshape(lm,(-1,2)).T
            _,lm_num = lm_temp.shape
            for ip in range(lm_num):
                pt = (int(lm_temp[0,ip]), int(lm_temp[1,ip]))
                cv2.circle(temp_rec, pt, 1, (255, 0, 0), 2)	
                cv2.putText(temp_rec, str(ip), pt, font, 0.6, (187, 255, 255), 1, cv2.LINE_AA)
        cv2.namedWindow('show_tensor', cv2.WINDOW_AUTOSIZE)
        cv2.imshow('show_tensor', cv2.resize(temp_rec[:,:,::-1],(1024,1024)))
        cv2.waitKey(0)     
        
        
def show_tensor_images(img,lm= None,batch=12,winname='show_tensor'):
    '''
    img: B x 3 x W x H
    lm: B x 2 x 68
    '''

    image_rec=(img).transpose(1,2).transpose(2,3).detach().cpu().numpy()
    
    for image_iter in range(batch):
        temp_rec = (image_rec[image_iter].copy()*255).astype(np.uint8)
        #temp_rec = Morphology_Closing(temp_rec,1)
        if not lm is None:
            lm_temp=lm[image_iter].detach().cpu().numpy()
            _,lm_num = lm_temp.shape
            for ip in range(lm_num):
                pt = (int(lm_temp[0,ip]), int(lm_temp[1,ip]))
                cv2.circle(temp_rec, pt, 1, (255, 0, 0), 2)	
                cv2.putText(temp_rec, str(ip), pt, font, 0.6, (187, 255, 255), 1, cv2.LINE_AA)
        cv2.namedWindow(winname, cv2.WINDOW_AUTOSIZE)
        cv2.imshow(winname, cv2.resize(temp_rec[:,:,::-1],(1024,1024)))
        cv2.waitKey(0)     
def torch_to_np_img(img):
	return cv2.cvtColor(img.detach().cpu().numpy().transpose(1, 2, 0) * 255, cv2.COLOR_BGR2RGB)

def write_image(image,filename):
	I = image.detach().to('cpu').numpy()#torch.clamp(image.detach(),min=0,max=1).to('cpu').numpy()
	Io = np.flip(np.swapaxes(np.swapaxes(I, 0, 1), 1, 2) * 255, 2)
	cv.imwrite(filename, Io)

def write_tiled_image(image,filename,tile_column_num):
	batch,ch,height,width = image.shape

	if batch%tile_column_num==0:
		tile_row_num = int(batch/tile_column_num)
	else:
		tile_row_num = int(batch / tile_column_num)+1
		image = torch.cat([image,torch.zeros(tile_row_num*tile_column_num-batch,ch,height,width,device=image.device)],dim=0)

	image = image.reshape(tile_row_num,tile_column_num,ch,height,width).transpose(1,2).transpose(2,3).reshape(tile_row_num,ch,height,tile_column_num*width)
	image = image.transpose(0,1).reshape(ch,tile_row_num*height,tile_column_num*width)

	write_image(image,filename)
