# TODO - 计算模型的推理时间

from architecture import *
from utils import *
import scipy.io as scio
import torch
import os
import numpy as np
from option import opt
from utils import *
from torch.backends import cudnn
import tqdm
from thop import profile
def calcTime():
    device = 'cuda:0'
    
    if opt.method == 'hdnet':
        model, FDL_loss = model_generator(opt.method, opt.pretrained_model_path)
        model = model
    else:
        model = model_generator(opt.method, opt.pretrained_model_path)
    cudnn.benchmark = True
    x = torch.rand(1,28,256,256).cuda()
    m = torch.rand(1,28,256,256).cuda()
    out = model(x,m)
    # model = model.to(device)
    repetitions = 1000
    
    total = sum(p.numel() for p in model.parameters())

    print("Total params: %.2fM" % (total/1e6))

    dummy_input1 = torch.rand(1,28,256,256).to(device)
    dummy_input2 = torch.rand(1,28,256,256).to(device)
    flops, params = profile(model, inputs=(dummy_input1,dummy_input2,)) 
    # # model_out, loss = model(y, Phi, Phi_sum)
    # print(1)
    print('FLOPs = ' + str(flops/1000**3) + 'G')
    print('Params = ' + str(params/1000**2) + 'M')


if __name__ == '__main__':
    calcTime()