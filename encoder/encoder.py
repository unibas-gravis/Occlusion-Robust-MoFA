import torchvision
import torch
import torch.nn as nn
import torch.nn.init as init
import encoder.camera_matrix as cam

class FaceEncoder(nn.Module):
    def __init__(self,obj,net='res50'):
        super(FaceEncoder, self).__init__()
        self.net=net
        
        self.vgg = torchvision.models.resnet50(pretrained=True)
        init.constant_(self.vgg.fc.bias, 0)
        init.constant_(self.vgg.fc.weight, 0)
            
        #self.vgg = torchvision.models.vgg19(pretrained=True)
        #init.constant_(self.vgg.classifier[6].bias, 0)
        #init.constant_(self.vgg.classifier[6].weight, 0)

        self.shape_dim = obj.shape_dim
        self.shape_idx = obj.shape_dim
        self.exp_idx = self.shape_idx + obj.exp_dim
        self.color_idx = self.exp_idx + obj.color_dim
        self.camera_idx = self.color_idx + 6
        self.light_idx = self.camera_idx + 27

    def forward(self, image):

        feature = self.vgg(image)
        shape_param = feature[:, :self.shape_idx]
        exp_param = feature[:,self.shape_idx:self.exp_idx]
        color_param = feature[:, self.exp_idx:self.color_idx]
        camera_param = feature[:, self.color_idx:self.camera_idx] 
        sh_coef = feature[:,self.camera_idx:self.light_idx]

        return shape_param, exp_param, color_param,camera_param,sh_coef

def convert_params(shape_param,exp_param,color_param,camera_param,sh_param,obj,T_ini,sh_ini,camera2world=True):

    vertex = obj.shape_mean.view(1, 3, -1) + torch.sum(obj.shape_basis.view(1, obj.shape_dim, 3, -1) * shape_param.view(-1, obj.shape_dim, 1, 1), dim=1)
    vertex += obj.exp_mean.view(1, 3, -1) + torch.sum(obj.exp_basis.view(1, obj.exp_dim, 3, -1) * exp_param.view(-1, obj.exp_dim, 1, 1), dim=1)

    color = obj.color_mean.view(1, 3, -1) + torch.sum(obj.color_basis.view(1, obj.color_dim, 3, -1) * color_param.view(-1, obj.color_dim, 1, 1), dim=1)

    R, T = cam.calc_extrinxic_camera_matrix(camera_param, T_ini,camera2world)

    sh_coef = sh_param + sh_ini.unsqueeze(0)

    return vertex, color, R, T, sh_coef


def convert_params_noexp(shape_param,exp_param,color_param,camera_param,sh_param,obj,T_ini,sh_ini,camera2world=True):

    vertex = obj.shape_mean.view(1, 3, -1) + torch.sum(obj.shape_basis.view(1, obj.shape_dim, 3, -1) * shape_param.view(-1, obj.shape_dim, 1, 1), dim=1)
    #vertex += obj.exp_mean.view(1, 3, -1) + torch.sum(obj.exp_basis.view(1, obj.exp_dim, 3, -1) * exp_param.view(-1, obj.exp_dim, 1, 1), dim=1)
    color = obj.color_mean.view(1, 3, -1) + torch.sum(obj.color_basis.view(1, obj.color_dim, 3, -1) * color_param.view(-1, obj.color_dim, 1, 1), dim=1)

    R, T = cam.calc_extrinxic_camera_matrix(camera_param, T_ini,camera2world)

    sh_coef = sh_param + sh_ini.unsqueeze(0)

    return vertex, color, R, T, sh_coef
