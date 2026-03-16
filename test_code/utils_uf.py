from operator import mod
from numpy import random
import scipy.io as sio
import os 
import numpy as np
import matplotlib.pyplot as plt
import math
import torch 
import logging

from ssim_torch import ssim

def get_Phi(mask_path, step=2, num_chan=28, batch_size=1, mask_crop_size=256):
    mask = sio.loadmat(mask_path + '/mask.mat')['mask']
    mask = mask[:mask_crop_size, :mask_crop_size]
    h, w = mask.shape
    Phi = np.zeros((1, num_chan, h, w+step*(num_chan-1)), dtype=np.float32)
    for i in range(num_chan):
        Phi[0, i, :, i*step:i*step+w] = mask
    Phi = torch.tensor(Phi, dtype=torch.float32)
    Phi_sum = torch.sum(Phi**2, dim=1)
    Phi_sum[Phi_sum==0] = 1
    return Phi.repeat(batch_size,1,1,1), Phi_sum.repeat(batch_size,1,1)

def LoadTraining(path):
    imgs = []
    scene_list = os.listdir(path)
    scene_list.sort()
    print('training sences:', len(scene_list))
    max_ = 0
    for i in range(len(scene_list)):
        scene_path = path + scene_list[i]
        if 'mat' not in scene_path:
            continue
        img_dict = sio.loadmat(scene_path)
        if "img_expand" in img_dict:
            img = img_dict['img_expand']/65536.
        elif "img" in img_dict:
            img = img_dict['img']/65536.
        img = img.astype(np.float32)
        imgs.append(img)
        print('Sence {} is loaded. {}'.format(i, scene_list[i]))

    return imgs

def LoadTest(path_test):
    scene_list = os.listdir(path_test)
    scene_list.sort()
    # scene_list =scene_list[5:]
    test_data = np.zeros((len(scene_list), 256, 256, 28))
    for i in range(len(scene_list)):
        scene_path = path_test + scene_list[i]
        img = sio.loadmat(scene_path)['img']
        #img = img/img.max()
        test_data[i,:,:,:] = img
        print(i, img.shape, img.max(), img.min())
    test_data = torch.from_numpy(np.transpose(test_data, (0, 3, 1, 2)))
    return test_data

def psnr(img1, img2):
    psnr_list = []
    for i in range(img1.shape[0]):
        total_psnr = 0
        #PIXEL_MAX = img2.max()
        PIXEL_MAX = img2[i,:,:,:].max()
        for ch in range(28):
            mse = np.mean((img1[i,:,:,ch] - img2[i,:,:,ch])**2)
            total_psnr += 20 * math.log10(PIXEL_MAX / math.sqrt(mse))
        psnr_list.append(total_psnr/img1.shape[3])
    return psnr_list

# def torch_psnr(img, ref):      #input [28,256,256]
#     nC = img.shape[0]
#     pixel_max = torch.max(ref)
#     psnr = 0
#     for i in range(nC):
#         mse = torch.mean((img[i,:,:] - ref[i,:,:]) ** 2)
#         psnr += 20 * torch.log10(pixel_max / torch.sqrt(mse))
#     return psnr/nC

def torch_psnr(img, ref):  # input [28,256,256]
    img = (img*256).round()
    ref = (ref*256).round()
    nC = img.shape[0]
    psnr = 0
    for i in range(nC):
        mse = torch.mean((img[i, :, :] - ref[i, :, :]) ** 2)
        psnr += 10 * torch.log10((255*255)/mse)
    return psnr / nC


def torch_ssim(img, ref):   #input [28,256,256]
    return ssim(torch.unsqueeze(img,0), torch.unsqueeze(ref,0))


def time2file_name(time):
    year = time[0:4]
    month = time[5:7]
    day = time[8:10]
    hour = time[11:13]
    minute = time[14:16]
    second = time[17:19]
    time_filename = year + '_' + month + '_' + day + '_' + hour + '_' + minute + '_' + second
    return time_filename

def augment_img(img, mode=0):
    if mode == 0:
        return img
    elif mode == 1:
        return torch.rot90(img, 1, [2,3])
    elif mode == 2:
        return torch.rot90(img, -1, [2,3])
    elif mode == 3:
        return torch.fliplr(img)
    elif mode == 4:
        return torch.flipud(img)
    # elif mode == 5:
    #     return np.rot90(img)
    # elif mode == 6:
    #     return np.rot90(img, k=2)
    # elif mode == 7:
    #     return np.flipud(np.rot90(img, k=3))

def shuffle_crop(train_data, batch_size, crop_size=256):
    
    index = np.random.choice(range(len(train_data)), batch_size)
    processed_data = np.zeros((batch_size, crop_size, crop_size, 28), dtype=np.float32)
    
    for i in range(batch_size):
        h, w, _ = train_data[index[i]].shape
        x_index = np.random.randint(0, h - crop_size)
        y_index = np.random.randint(0, w - crop_size)
        processed_data[i, :, :, :] = train_data[index[i]][x_index:x_index + crop_size, y_index:y_index + crop_size, :] 
    gt_batch = torch.from_numpy(np.transpose(processed_data, (0, 3, 1, 2))).type(torch.float32)
    # mode = random.randint(5)
    # mode = 0
    # gt_batch = augment_img(gt_batch.cuda(), mode=mode)
    return gt_batch

def shift(inputs, step=2, mode='torch:(b,c,h,w)->(b,c,h,w+s)'):
    '''
    mode(str): 
        torch:(b,c,h,w)->(b,c,h,w+s) (default)
        numpy:(h,w,c)->(h,w+s,c)
    '''
    if mode == 'torch:(b,c,h,w)->(b,c,h,w+s)':
        b, c, h, w = inputs.shape
        output = torch.zeros(b, c, h, w+(c-1)*step, dtype=torch.float32, device=inputs.device)
        for i in range(c):
            output[:,i,:,step*i:step*i+w] = inputs[:,i,:,:]
    elif mode == 'numpy:(h,w,c)->(h,w+s,c)':
        h, w, c = inputs.shape
        output = np.zeros((h, w+(c-1)*step, c), dtype=np.float32)
        for i in range(c):
            output[:,step*i:step*i+w,i] = inputs[:,:,i]
    return output


def window_loss(out,gt,window_size):
    loss_win_list=[]
    b,c,h,w = gt.shape
    for i in range(h//window_size):
        for j in range(w//window_size):
            out_win = out[:,:,i*window_size:(i+1)*window_size,j*window_size:(j+1)*window_size]
            gt_win = gt[:,:,i*window_size:(i+1)*window_size,j*window_size:(j+1)*window_size]
            loss_win = torch.sqrt(torch.mean((out_win-gt_win)**2))
            loss_win_list.append(loss_win)
    return loss_win_list  

def adaptive_boost(out,gt,window_size):
    loss_win_list = window_loss(out,gt,window_size)
    loss_win_list_norm =[torch.exp(loss_win_num/torch.sum(torch.stack(loss_win_list))) for loss_win_num in loss_win_list]
    return loss_win_list,loss_win_list_norm


def shift_back(inputs, step=2, channel=28, mode='torch:(b,h,w+s)->(b,c,h,w)'):
    '''
    mode(str): 
        torch:(b,h,w+s)->(b,c,h,w) (default)
        torch:(b,c,h,w+s)->(b,c,h,w)
        numpy:(h,w+s)->(h,w,c)
    '''
    if mode == 'torch:(b,h,w+s)->(b,c,h,w)':
        b, h, w_ = inputs.shape
        c = channel
        output = torch.zeros(b, c, h, w_-(c-1)*step, dtype=torch.float32, device=inputs.device)
        for i in range(c):
            output[:,i,:,:] = inputs[:,:,step*i:step*i+w_-(c-1)*step]
    elif mode == 'torch:(b,c,h,w+s)->(b,c,h,w)':
        b, c, h, w_ = inputs.shape
        output = torch.zeros(b, c, h, w_-(c-1)*step, dtype=torch.float32, device=inputs.device)
        for i in range(c):
            output[:,i,:,:] = inputs[:,i,:,step*i:step*i+w_-(c-1)*step]
    elif mode == 'numpy:(h,w+s)->(h,w,c)':
        h, w_ = inputs.shape
        c = channel
        output = np.zeros((h, w_-(c-1)*step, c), dtype=np.float32)
        for i in range(c):
            output[:,:,i] = inputs[:,step*i:step*i+w_-(c-1)*step]
    return output

def gen_log(model_path):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO) 
    formatter = logging.Formatter("%(asctime)s - %(levelname)s: %(message)s")
    
    log_file = model_path + '/log.txt'
    fh = logging.FileHandler(log_file, mode='a')
    fh.setLevel(logging.INFO) 
    fh.setFormatter(formatter)

    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)
    
    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger

def A(x, Phi):
    return torch.sum(x*Phi, dim=1)  # element-wise product

# def At(y, Phi):
#     y_ = y.unsqueeze(1) if y.ndim==3 else y
#     return y_ * Phi

def At(y,Phi):
    temp = torch.unsqueeze(y, 1).repeat(1,Phi.shape[1],1,1)
    x = temp*Phi
    return x
