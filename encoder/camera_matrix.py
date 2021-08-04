import torch
import torch.nn as nn
import math

def euler_rotation(rot_param):
    rx_mul = torch.Tensor([[0, 0, 0, 0, 1, 1, 0, 1, 1]]).to(rot_param.device)
    rx_add = torch.Tensor([[0, 1, 1, 1, 0, 1, 1, 3, 0]]).to(rot_param.device) * math.pi / 2.0
    ry_mul = torch.Tensor([[1, 0, 1, 0, 0, 0, 1, 0, 1]]).to(rot_param.device)
    ry_add = torch.Tensor([[0, 1, 3, 1, 0, 1, 1, 1, 0]]).to(rot_param.device) * math.pi / 2.0
    rz_mul = torch.Tensor([[1, 1, 0, 1, 1, 0, 0, 0, 0]]).to(rot_param.device)
    rz_add = torch.Tensor([[0, 1, 1, 3, 0, 1, 1, 1, 0]]).to(rot_param.device) * math.pi / 2.0

    Rx =  torch.cos(rx_mul*rot_param[:,0:1]+rx_add)
    Ry =  torch.cos(ry_mul*rot_param[:,1:2]+ry_add)
    Rz =  torch.cos(rz_mul*rot_param[:,2:3]+rz_add)
    R = Rx.view(-1,3,3)@Ry.view(-1,3,3)@Rz.view(-1,3,3)
    return R

def calc_extrinxic_camera_matrix(param, T_ini, camera2world=True):

    Rb = torch.eye(3, 3, device=param.device)
    Rb = Rb.view(1, 3, 3)
    Rb[0, 1, 1] = -1
    Rb[0, 2, 2] = -1

    R = Rb @ euler_rotation(param[:, :3]) 
    T = (param[:, 3:6] + T_ini).view(-1, 3, 1)

    if camera2world == True:
        R = R.transpose(1, 2)
        T = -R @ T

    return R, T