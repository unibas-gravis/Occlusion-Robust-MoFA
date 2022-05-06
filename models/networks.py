from .backbones import get_model
import torch.nn as nn
import torch
import torch.nn.functional as F
from kornia.geometry import warp_affine
from util.preprocess import estimate_norm_torch
import util.util as util
def resize_n_crop(image, M, dsize=112):
    # image: (b, c, h, w)
    # M   :  (b, 2, 3)
    return warp_affine(image, M, dsize=(dsize, dsize))


def define_net_recog(net_recog, pretrained_path=None):
    net = RecogNetWrapper(net_recog=net_recog, pretrained_path=pretrained_path)
    net.eval()
    return net


class RecogNetWrapper(nn.Module):
    def __init__(self, net_recog, pretrained_path=None, input_size=112):
        super(RecogNetWrapper, self).__init__()
        net = get_model(name=net_recog, fp16=False)
        if pretrained_path:
            state_dict = torch.load(pretrained_path, map_location='cpu')
            net.load_state_dict(state_dict)
            print("loading pretrained net_recog %s from %s" %(net_recog, pretrained_path))
        for param in net.parameters():
            param.requires_grad = False
        self.net = net
        self.preprocess = lambda x: 2 * x - 1
        self.input_size=input_size
        
    def forward(self, image,pred_lm,is_shallow=False):

        M = estimate_norm_torch(pred_lm, 224)
        resized=resize_n_crop(image, M, self.input_size)
        #util.show_tensor_images(resized,batch=12)
        image = self.preprocess(resized)
        feature = self.net(image,is_shallow)
        id_feature = F.normalize(feature, dim=-1, p=2)
        return id_feature
