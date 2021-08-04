import torch
import numpy as np
import sys
import os

device_ids=[0,1,2,3]


def write_obj_with_colors(obj_name, vertices, triangles, colors):
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
            s = 'v {} {} {} {} {} {}\n'.format(vertices[i, 0], vertices[i, 1], vertices[i, 2], colors[i, 0], colors[i, 1], colors[i, 2])
            f.write(s)

        [k, ntri] = triangles.shape
        for i in range(triangles.shape[0]):
            s = 'f {} {} {}\n'.format(triangles[i, 2], triangles[i, 1], triangles[i, 0])
            f.write(s)
            
            

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

    
