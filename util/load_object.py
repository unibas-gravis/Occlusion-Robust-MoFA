import h5py
import numpy as np
import math
import torch
import csv
import torch.nn as nn
import torchvision.transforms.functional as tf

class Object3DMM():

    def __init__(self,filename, device):

        self.point_num = 0
        self.face_num = 0
        self.shape_dim = 0
        self.exp_dim = 0
        self.color_dim = 0
        self.face = None
        self.shape_mean=None
        self.shape_basis = None
        self.exp_mean=None
        self.exp_basis = None
        self.color_mean=None
        self.color_basis = None

        data = h5py.File(filename,'r')
        
        reference_mesh = np.array(data['shape']['representer']['points'])
        
        self.point_num = np.array(data['shape']['representer']['points']).shape[1]
        self.face = torch.LongTensor(np.array(data['shape']['representer']['cells']))
        self.face_num = self.face.shape[1]
        self.shape_mean = torch.Tensor(np.array(data['shape']['model']['mean']).reshape(self.point_num,3).T)
        self.shape_std = np.sqrt(np.array(data['shape']['model']['pcaVariance']))
        self.shape_basis = (np.array(data['shape']['model']['pcaBasis'])*np.sqrt(np.array(data['shape']['model']['pcaVariance']))).T.reshape(-1,self.point_num,3)
        self.shape_basis = torch.Tensor(np.swapaxes(self.shape_basis,1,2))
        self.exp_mean = torch.Tensor(np.array(data['expression']['model']['mean']).reshape(self.point_num,3).T)
        self.exp_std = np.sqrt(np.array(data['expression']['model']['pcaVariance']))
        self.exp_basis = (np.array(data['expression']['model']['pcaBasis'])*np.sqrt(np.array(data['expression']['model']['pcaVariance']))).T.reshape(-1,self.point_num,3)
        self.exp_basis = torch.Tensor(np.swapaxes(self.exp_basis, 1, 2))
        self.color_mean = torch.Tensor(np.array(data['color']['model']['mean']).reshape(self.point_num,3).T)
        self.color_std = np.sqrt(np.array(data['color']['model']['pcaVariance']))
        self.color_basis = (np.array(data['color']['model']['pcaBasis'])*np.sqrt(np.array(data['color']['model']['pcaVariance']))).T.reshape(-1,self.point_num,3)
        self.color_basis = torch.Tensor(np.swapaxes(self.color_basis, 1, 2))
        self.shape_dim = self.shape_basis.shape[0]
        self.exp_dim = self.exp_basis.shape[0]
        self.color_dim = self.color_basis.shape[0]
        
        
        landmark_idxs = [22212, 23003, 23412, 23817, 44075, 45617, 46739, 47456, 47847, 48237, 48953, 50064, 51700, 30656, 31279, 31775, 32532,\
                         38719,39462,39835,40033,40252,40668,40882,\
                         41081,41459,42172,8124,8140,8149,8155,6750,\
                             7459,8171,8993,9699,2735,4275,5050,6088,\
                                 4804,4030,10263,11159,11932,13746,12203,11429,5394,6283,\
                                     7474,8183,9007,10203,11116,9380,8673,8201,7847,7020,6539,7600,8190,8780,10339,8545,8191,7837]
                        

        self.landmark = torch.LongTensor(landmark_idxs).to(device)
        self.shape_mean = self.shape_mean.to(device)
        self.shape_basis = self.shape_basis.to(device)
        self.exp_mean = self.exp_mean.to(device)
        self.exp_basis = self.exp_basis.to(device)
        self.color_mean = self.color_mean.to(device)
        self.color_basis = self.color_basis.to(device)
        self.crop_center()
        world_center = torch.Tensor([0, 0, 100]).to(device) #center of world coordinate in 3DMM space

        self.adjust_center(world_center)
        self.weight_lm = tf.to_tensor(np.expand_dims(np.asarray([0.2,0.2,0.2,0.2,0.2,\
                                                    0.2,0.2,0.2,0.2,0.2,\
                                                    0.2,0.2,\
                                                    0.2,0.2,0.2,0.2,0.2,\
                                                    0.1,0.1,0.1,0.1,0.1,\
                                                    0.1,0.1,0.1,0.1,0.1,\
                                                    1,1,1,1,1,1,1,1,1,\
                                                    1,1,1,1,1,1,1,1,1,1,1,1,\
                                                    1,1,1,1,1,1,1,1,1,1,1,1,\
                                                        1,1,1,1,1,1,1,1]),0)).to(device)

        data.close()

    def crop_center(self):
        device = self.shape_mean.device
        shape_mean_temp = self.shape_mean[0:2,:].clone()
            
        #(x/z/100)^2+((y)/z/0.75)^2 <=1
        pt = shape_mean_temp/self.shape_mean[2:3,:]
        c1 = (pt[0,:]**2/100**2 + pt[1,:]**2/0.75**2 <= 1)
        
        #||(x/z/100)^2+((y+18)/z/0.62)^2 <=1
        shape_mean_temp[1,:] += 18
        pt = shape_mean_temp/self.shape_mean[2:3,:]
        c2 = (pt[0,:]**2/100**2 + pt[1,:]**2/0.62**2 <= 1)
        
        #|| (x/z/3.2)^2+((y-25)/z/0.92)^2 <=1
        shape_mean_temp = self.shape_mean[0:2,:].clone()
        shape_mean_temp[1,:] -= 25
        pt = shape_mean_temp/self.shape_mean[2:3,:]
        c3 = (pt[0,:]**2/3.2**2 + pt[1,:]**2/0.92**2 <= 1)
        
        
        # && z+0.05*y>=20
        c4=self.shape_mean[2,:] + 0.05*self.shape_mean[1,:]>20
        vis_vert = (c1 | c2 | c3)*c4

        
        vis_face = vis_vert[self.face[0,:]]*vis_vert[self.face[1,:]]*vis_vert[self.face[2,:]]

        used_vert = torch.BoolTensor(self.point_num).to(device)
        used_vert[:] = False
        used_vert[self.face[:,vis_face]] = True

        vis_vert = vis_vert*used_vert
        idx_r2o = torch.LongTensor(range(0,self.point_num))[vis_vert].to(device)
        self.idx_r2o=idx_r2o
        idx_o2r = torch.LongTensor(range(0,self.point_num)).to(device)
        idx_o2r[idx_r2o] = torch.LongTensor(range(0,idx_r2o.shape[0])).to(device)
        self.idx_o2r = idx_o2r
        self.landmark = idx_o2r[self.landmark]

        self.point_num = idx_r2o.shape[0]
        self.shape_mean = self.shape_mean[:,vis_vert]
        self.shape_basis = self.shape_basis[:,:,vis_vert]
        self.exp_mean = self.exp_mean[:,vis_vert]
        self.exp_basis = self.exp_basis[:,:,vis_vert]
        self.color_mean = self.color_mean[:,vis_vert]
        self.color_basis = self.color_basis[:,:,vis_vert]

        self.face = idx_o2r[self.face[:,vis_face]]
        self.face_num = self.face.shape[1]

        print("vert num {0}".format(self.point_num))
        print("face num {0}".format(self.face_num))


    def adjust_center(self,center = None):

        if center is None:
            center = torch.mean(self.shape_mean + self.exp_mean,dim=1)
        self.shape_mean -= center.view(3,1)

    def save_ply(self,filename,shape_coeff,exp_coeff,color_coeff):
        shape_coeff = shape_coeff.to('cpu')
        exp_coeff = exp_coeff.to('cpu')
        color_coeff = color_coeff.to('cpu')

        out = open(filename, 'wb')
        out.write('ply\r\n'.encode('ascii'))
        out.write('format ascii 1.0\r\n'.encode('ascii'))
        out.write('element vertex {}\r\n'.format(self.point_num).encode('ascii'))
        out.write('property float x\r\n'.encode('ascii'))
        out.write('property float y\r\n'.encode('ascii'))
        out.write('property float z\r\n'.encode('ascii'))
        out.write('property uchar red\r\n'.encode('ascii'))
        out.write('property uchar green\r\n'.encode('ascii'))
        out.write('property uchar blue\r\n'.encode('ascii'))
        out.write('element face {}\r\n'.format(self.face_num).encode('ascii'))
        out.write('property list uchar uint vertex_indices\r\n'.encode('ascii'))
        out.write('end_header\r\n'.encode('ascii'))

        shape = self.shape_mean + torch.sum(shape_coeff*self.shape_basis,0)+ self.exp_mean + torch.sum(exp_coeff*self.exp_basis,0)
        color = torch.clamp(self.color_mean + torch.sum(color_coeff*self.color_basis,0),min=0,max=1)

        for n in range(0, self.point_num):
            s = '{} {} {} {} {} {}\r\n'.format(shape[0,n], shape[1,n], shape[2,n],
                                               math.floor(color[0,n] * 255), math.floor(color[1,n] * 255),
                                               math.floor(color[2,n] * 255)).encode('ascii')

            out.write(s)

        for n in range(0, self.face_num):
            s = '3 {} {} {}\r\n'.format(self.face[0][n], self.face[1][n], self.face[2][n]).encode('ascii')
            out.write(s)

        out.close()
