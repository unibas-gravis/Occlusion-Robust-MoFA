import h5py
import numpy as np
import math
import torch
import csv
import torch.nn as nn
import torchvision.transforms.functional as tf
import os
import pickle
'''--------------

68 2D landmarks used

--------------'''
class Object3DMM():

    def __init__(self,filename, device,is_crop = False):

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
        if is_crop:
            current_path = os.getcwd() 
            info_path = current_path+'/basel_3DMM/idxes_crop.pkl'
            idxes = pickle.load(open(info_path,'rb'))
            arr_points = data['shape']['representer']['points'][()].take(idxes['points'],axis=1)
            cells = idxes['cells']
            shape_mean = data['shape']['model']['mean'][()].take(idxes['shape_mean'],axis=0)
            expression_mean = data['expression']['model']['mean'][()].take(idxes['exp_mean'],axis=0)
            color_mean = data['color']['model']['mean'][()].take(idxes['color_mean'],axis=0)
            shape_pca = data['shape']['model']['pcaBasis'][()].take(idxes['shape_pcaBasis'],axis=0)
            expression_pca = data['expression']['model']['pcaBasis'][()].take(idxes['exp_pcaBasis'],axis=0)
            color_pca = data['color']['model']['pcaBasis'][()].take(idxes['color_pcaBasis'],axis=0)
            self.point_num =arr_points.shape[1]
            self.face = torch.LongTensor(cells).to(device)
            self.face_num = self.face.shape[1]
            self.shape_mean = torch.Tensor(shape_mean.reshape(self.point_num,3).T)
            #self.shape_pca = np.array(data['shape']['model']['pcaBasis'])
            self.shape_std = np.sqrt(np.array(data['shape']['model']['pcaVariance']))
            self.shape_basis = (shape_pca*np.sqrt(np.array(data['shape']['model']['pcaVariance']))).T.reshape(-1,self.point_num,3)
            self.shape_basis = torch.Tensor(np.swapaxes(self.shape_basis,1,2))
            self.exp_mean = torch.Tensor(expression_mean.reshape(self.point_num,3).T)
            self.exp_std = np.sqrt(np.array(data['expression']['model']['pcaVariance']))
            self.exp_basis = (expression_pca*np.sqrt(np.array(data['expression']['model']['pcaVariance']))).T.reshape(-1,self.point_num,3)
            self.exp_basis = torch.Tensor(np.swapaxes(self.exp_basis, 1, 2))
            self.color_mean = torch.Tensor(color_mean.reshape(self.point_num,3).T)
            self.color_std = np.sqrt(np.array(data['color']['model']['pcaVariance']))
            self.color_basis = (color_pca*np.sqrt(np.array(data['color']['model']['pcaVariance']))).T.reshape(-1,self.point_num,3)
            self.color_basis = torch.Tensor(np.swapaxes(self.color_basis, 1, 2))
            
            
        else:
            
            self.point_num = np.array(data['shape']['representer']['points']).shape[1]
            self.face = torch.LongTensor(np.array(data['shape']['representer']['cells'])).to(device)
            self.face_num = self.face.shape[1]
            self.shape_mean = torch.Tensor(np.array(data['shape']['model']['mean']).reshape(self.point_num,3).T)
            #self.shape_pca = np.array(data['shape']['model']['pcaBasis'])
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
        
                        
        landmark_idxs = [16678,16817,16480,16254,32170,32866,33302,\
                         33598,33765,33949,34239,34693,35398,27717,27536,\
                             27137,27249,\
                                 27998,28747,29133,29336,29503,30188,30352,\
                                     30557,30947,31682,8122,8137,8147,8152,6627,\
                                     7219,8165,9107,9695,2475,4275,5050,5829,4802,3900,\
                                         10382,11280,12053,14123,12323,11419,5137,5897,7467,8175,\
                                                 8883,10450,11110,9372,8666,8194,7722,\
                                                 7012,5524,7358,8184,9010,10852,8659,8187,7715]
        ringnet_lm_idxs = [2735, 6088, 10263, 13746,8165,5394 ,11116] # 7 lm for full face
        self.ringnet_lm = torch.LongTensor(ringnet_lm_idxs).to(device)
        self.landmark = torch.LongTensor(landmark_idxs).to(device)
        self.shape_mean = self.shape_mean.to(device)
        self.shape_basis = self.shape_basis.to(device)
        self.exp_mean = self.exp_mean.to(device)
        self.exp_basis = self.exp_basis.to(device)
        self.color_mean = self.color_mean.to(device)
        self.color_basis = self.color_basis.to(device)
        #self.crop_center()
        world_center = torch.Tensor([0, 0, 100]).to(device) #center of world coordinate in 3DMM space
        self.adjust_center(world_center)
        weight = np.ones([68])
        weight[28:31] = 20
        weight[-8:] = 20
        weight = np.expand_dims(weight, 0)
        self.weight_lm = torch.tensor(weight).to(device)

        data.close()

    
    def adjust_center(self,center = None):

        if center is None:
            center = torch.mean(self.shape_mean + self.exp_mean,dim=1)
        self.shape_mean -= center.view(3,1)

    
