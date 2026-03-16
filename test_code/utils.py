import scipy.io as sio
import os
import numpy as np
import torch
import logging
import h5py
from ssim_torch import ssim
def torch_psnr(img, ref):  # input [28,256,256]
    img = (img*256).round()
    ref = (ref*256).round()
    nC = img.shape[0]
    psnr = 0
    for i in range(nC):
        mse = torch.mean((img[i, :, :] - ref[i, :, :]) ** 2)
        psnr += 10 * torch.log10((255*255)/mse)
    return psnr / nC

def torch_ssim(img, ref):  # input [28,256,256]
    return ssim(torch.unsqueeze(img, 0), torch.unsqueeze(ref, 0))


def generate_masks(mask_path, batch_size):
    mask = sio.loadmat(mask_path + '/mask.mat')
    mask = mask['mask']
    mask3d = np.tile(mask[:, :, np.newaxis], (1, 1, 28))
    mask3d = np.transpose(mask3d, [2, 0, 1])
    mask3d = torch.from_numpy(mask3d)
    [nC, H, W] = mask3d.shape
    mask3d_batch = mask3d.expand([batch_size, nC, H, W]).cuda().float()
    return mask3d_batch

# def generate_shift_masks(mask_path, batch_size):
#     mask = sio.loadmat(mask_path + '/mask_3d_shift.mat')
#     mask_3d_shift = mask['mask_3d_shift']
#     mask_3d_shift = np.transpose(mask_3d_shift, [2, 0, 1])
#     mask_3d_shift = torch.from_numpy(mask_3d_shift)
#     [nC, H, W] = mask_3d_shift.shape
#     Phi_batch = mask_3d_shift.expand([batch_size, nC, H, W]).cuda().float()
#     Phi_s_batch = torch.sum(Phi_batch**2,1)
#     Phi_s_batch[Phi_s_batch==0] = 1
#     # print(Phi_batch.shape, Phi_s_batch.shape)
#     return Phi_batch, Phi_s_batch

def generate_shift_masks(mask_path, batch_size):
    mask = sio.loadmat(mask_path + '/mask.mat')
    mask = mask['mask']
    mask3d = np.tile(mask[:, :, np.newaxis], (1, 1, 28))
    mask3d = np.transpose(mask3d, [2, 0, 1])
    mask3d = torch.from_numpy(mask3d)
    mask_3d_shift=mask3d
    [nC, H, W] = mask_3d_shift.shape
    Phi_batch = mask_3d_shift.expand([batch_size, nC, H, W]).cuda().float()
    Phi_batch = shift(Phi_batch)
    Phi_s_batch = torch.sum(Phi_batch**2,1)
    Phi_s_batch[Phi_s_batch==0] = 1
    # print(Phi_batch.shape, Phi_s_batch.shape)
    return Phi_batch, Phi_s_batch

def LoadTest(path_test):
    scene_list = os.listdir(path_test)
    scene_list.sort()
    test_data = np.zeros((len(scene_list), 256, 256, 28))
    for i in range(len(scene_list)):
        scene_path = path_test + scene_list[i]
        img = sio.loadmat(scene_path)['img']
        test_data[i, :, :, :] = img
    test_data = torch.from_numpy(np.transpose(test_data, (0, 3, 1, 2)))
    return test_data

def LoadTest_Harvard(path_test):
    scene_list = os.listdir(path_test)
    scene_list.sort()
    test_data = np.zeros((len(scene_list), 256, 256, 28))
    for i in range(len(scene_list)):
        scene_path = path_test + scene_list[i]
        img = sio.loadmat(scene_path)['ref']
       
        img = (img - np.min(img))/(np.max(img)-np.min(img))
        # img = h5py.File(scene_path1)['rgb']  # rad,rgb bands  icvl
        # aa = img.keys()
        # img = np.array(img)/4095
        img = img[:,:,1:29]
        img = img[200:456,0:256,:]
        test_data[i, :, :, :] = img
    test_data = torch.from_numpy(np.transpose(test_data, (0, 3, 1, 2)))
    return test_data


def LoadTest_ICVL(path_test):
    scene_list = os.listdir(path_test)
    scene_list.sort()
    test_data = np.zeros((len(scene_list), 256, 256, 28))
    for i in range(len(scene_list)):
        scene_path = path_test + scene_list[i]
        img = h5py.File(scene_path)['rad']
        img = np.array(img)/4095
        # img = (img - np.min(img))/(np.max(img)-np.min(img))
        # img = h5py.File(scene_path1)['rgb']  # rad,rgb bands  icvl
        # aa = img.keys()
        # img = np.array(img)/4095
        img = img[1:29,:,:]
        img = np.transpose(img,(2,1,0))
        img = img[400:656,100:356,:]
        test_data[i, :, :, :] = img
    test_data = torch.from_numpy(np.transpose(test_data, (0, 3, 1, 2)))
    return test_data




def LoadMeasurement(path_test_meas):
    img = sio.loadmat(path_test_meas)['simulation_test']
    test_data = img
    test_data = torch.from_numpy(test_data)
    return test_data

def time2file_name(time):
    year = time[0:4]
    month = time[5:7]
    day = time[8:10]
    hour = time[11:13]
    minute = time[14:16]
    second = time[17:19]
    time_filename = year + '_' + month + '_' + day + '_' + hour + '_' + minute + '_' + second
    return time_filename

def shuffle_crop(train_data, batch_size, crop_size=256):
    index = np.random.choice(range(len(train_data)), batch_size)
    processed_data = np.zeros((batch_size, crop_size, crop_size, 28), dtype=np.float32)
    for i in range(batch_size):
        h, w, _ = train_data[index[i]].shape
        x_index = np.random.randint(0, h - crop_size)
        y_index = np.random.randint(0, w - crop_size)
        processed_data[i, :, :, :] = train_data[index[i]][x_index:x_index + crop_size, y_index:y_index + crop_size, :]
    gt_batch = torch.from_numpy(np.transpose(processed_data, (0, 3, 1, 2)))
    return gt_batch
import matplotlib.pyplot as plt

def add_gaussian_noise(image, sigma):
    """
    添加高斯噪声到图像
    :param image: 原始图像，二维数组
    :param sigma: 高斯噪声的标准差
    :return: 添加噪声后的图像
    """
    noise = np.random.normal(0, sigma, image.shape)
    noisy_image = image + noise
    noisy_image_clipped = np.clip(noisy_image, 0, 1)  # 确保像素值在[0, 1]范围内
    return noisy_image_clipped

def gen_meas_torch(data_batch, mask3d_batch,  Y2H=True, mul_mask=False):
    [batch_size, nC, H, W] = data_batch.shape
    mask3d_batch = (mask3d_batch[0, :, :, :]).expand([batch_size, nC, H, W]).cuda().float()  # [10,28,256,256]
    temp = shift(mask3d_batch * data_batch, 2)
    meas = torch.sum(temp, 1)

    # input = meas.detach().cpu().numpy()
    # input = add_gaussian_noise(input,0.1)
    # input = torch.FloatTensor(input.copy())
    # input = input.cuda()
    # meas = input
    # for i in range(10):
    #     aa = meas[i,:,:]
    #     aa = aa.detach().cpu().numpy()
    #     plt.figure(1)
    #     plt.imshow(aa,cmap='gray')
    #     # plt.savefig('measure_4.png')
    #     plt.savefig(f'icvl_crop_256/mea_{i}.png')
    if Y2H:
        meas = meas / nC * 2
        aa = torch.max(meas)
        H = shift_back(meas)
        if mul_mask:
            HM = torch.mul(H, mask3d_batch)
            return HM
        return H
    return meas

def shift(inputs, step=2):
    [bs, nC, row, col] = inputs.shape
    output = torch.zeros(bs, nC, row, col + (nC - 1) * step).cuda().float()
    for i in range(nC):
        output[:, i, :, step * i:step * i + col] = inputs[:, i, :, :]
    return output

def shift_back(inputs, step=2):  # input [bs,256,310]  output [bs, 28, 256, 256]
    [bs, row, col] = inputs.shape
    nC = 28
    output = torch.zeros(bs, nC, row, col - (nC - 1) * step).cuda().float()
    for i in range(nC):
        output[:, i, :, :] = inputs[:, :, step * i:step * i + col - (nC - 1) * step]
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

def init_mask(mask_path, mask_type, batch_size):
    mask3d_batch = generate_masks(mask_path, batch_size)
    if mask_type == 'Phi':
        shift_mask3d_batch = shift(mask3d_batch)
        input_mask = shift_mask3d_batch
    elif mask_type == 'Phi_PhiPhiT':
        Phi_batch, Phi_s_batch = generate_shift_masks(mask_path, batch_size)
        input_mask = (Phi_batch, Phi_s_batch)
    elif mask_type == 'Mask':
        input_mask = mask3d_batch
    elif mask_type == None:
        input_mask = None
    return mask3d_batch, input_mask

def init_meas(gt, mask, input_setting):
    if input_setting == 'H':
        input_meas = gen_meas_torch(gt, mask, Y2H=True, mul_mask=False)
    elif input_setting == 'HM':
        input_meas = gen_meas_torch(gt, mask, Y2H=True, mul_mask=True)
    elif input_setting == 'Y':
        input_meas = gen_meas_torch(gt, mask, Y2H=False, mul_mask=True)
    return input_meas

def torch_sam(img,ref):
    a = torch.sum(torch.mul(img,ref),0)/(torch.norm(img,dim=0) * torch.norm(ref,dim=0)+1e-5)
    # sam = torch.mean(a)
    sam_jd = torch.arccos(a)
    sam_jd = sam_jd*180/3.1415926535
    return torch.mean(sam_jd)

def calc_ergas(img_fus,img_tgt):
    # bchw
    # img_tgt = np.squeeze(img_tgt)
    # img_fus = np.squeeze(img_fus)
    img_tgt = img_tgt.detach().cpu().numpy()
    img_fus = img_fus.detach().cpu().numpy()
    img_tgt = img_tgt.reshape(img_tgt.shape[0], -1)
    img_fus = img_fus.reshape(img_fus.shape[0], -1)

    rmse = np.mean((img_tgt-img_fus)**2, axis=1)
    rmse = rmse**0.5
    mean = np.mean(img_tgt, axis=1)

    ergas = np.mean((rmse/mean)**2)
    ergas = 100*ergas**0.5

    return ergas

# def calc_psnr(img_tgt, img_fus):
#     mse = np.mean((img_tgt-img_fus)**2)
#     img_max = np.max(img_tgt)
#     psnr = 10*np.log10(img_max**2/mse)

#     return psnr

# def calc_rmse(img_tgt, img_fus):
#     rmse = np.sqrt(np.mean((img_tgt-img_fus)**2))

#     return rmse

def calc_sam(img_fus,img_tgt):
    # img_tgt = np.squeeze(img_tgt)
    # img_fus = np.squeeze(img_fus)
    img_tgt = img_tgt.detach().cpu().numpy()
    img_fus = img_fus.detach().cpu().numpy()
    img_tgt = img_tgt.reshape(img_tgt.shape[0], -1)
    img_fus = img_fus.reshape(img_fus.shape[0], -1)
    img_tgt = img_tgt / np.max(img_tgt)
    img_fus = img_fus / np.max(img_fus)

    A = np.sqrt(np.sum(img_tgt**2, axis=0))
    B = np.sqrt(np.sum(img_fus**2, axis=0))
    AB = np.sum(img_tgt*img_fus, axis=0)

    sam = AB/(A*B)
    sam = np.arccos(sam)
    sam = np.mean(sam)*180/3.1415926535

    return sam


if __name__ =='__main__':
    img = torch.rand(28,256,256)
    ref = torch.rand(28,256,256)
    res = torch.sam(img,ref)