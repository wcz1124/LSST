from architecture import *
from utils import *
import scipy.io as scio
import torch
import os
import numpy as np
from option import opt
from focal_region_loss import *
os.environ["CUDA_DEVICE_ORDER"] = 'PCI_BUS_ID'
os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu_id
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
if not torch.cuda.is_available():
    raise Exception('NO GPU!')
from utils import *


mask3d_batch, input_mask = init_mask(opt.mask_path, opt.input_mask, 10)
criterion = FocalRegionLoss(loss_weight=1,alpha=0.5,log_matrix=True).cuda()
if not os.path.exists(opt.outf):
    os.makedirs(opt.outf)




def test(model):
    test_data = LoadTest(opt.test_path)
    test_gt = test_data.cuda().float()
    input_meas = init_meas(test_gt, mask3d_batch, opt.input_setting)
    model.eval()
    with torch.no_grad():
        model_out = model(input_meas, input_mask)   #10,28,256,256
    psnr_list, ssim_list , ergas_list, sam_list= [], [],[],[]
    for k in range(test_gt.shape[0]):
        psnr_val = torch_psnr(model_out[k,:,:,:], test_gt[k,:,:,:])
        ssim_val = torch_ssim(model_out[k,:,:,:], test_gt[k,:,:,:])
        ergas_val = calc_ergas(model_out[k,:,:,:], test_gt[k,:,:,:])
        sam_val = calc_sam(model_out[k,:,:,:], test_gt[k,:,:,:])
        psnr_list.append(psnr_val.detach().cpu().numpy())
        ssim_list.append(ssim_val.detach().cpu().numpy())
        ergas_list.append(ergas_val)
        sam_list.append(sam_val)
        # if (epoch % 100 == 0) and (writer is not None):
        #     writer.add_image(f'test_recon/{k}-image', Recon_img[k,13:14,:,:].cpu(), global_step=epoch)
    pred = np.transpose(model_out.detach().cpu().numpy(), (0, 2, 3, 1)).astype(np.float32)
    truth = np.transpose(test_gt.cpu().numpy(), (0, 2, 3, 1)).astype(np.float32)
    psnr_mean = np.mean(np.asarray(psnr_list))
    ssim_mean = np.mean(np.asarray(ssim_list))
    ergas_mean = np.mean(np.asarray(ergas_list))
    sam_mean = np.mean(np.asarray(sam_list))
    print(psnr_mean,ssim_mean,sam_mean,ergas_mean)
    # pred = np.transpose(model_out.detach().cpu().numpy(), (0, 2, 3, 1)).astype(np.float32)
    # truth = np.transpose(test_gt.cpu().numpy(), (0, 2, 3, 1)).astype(np.float32)
  
    return pred,truth,psnr_mean,ssim_mean

def main():
    # model
    if opt.method == 'hdnet':
        model, FDL_loss = model_generator(opt.method, opt.pretrained_model_path)
        model = model.cuda()
    else:
        model = model_generator(opt.method, opt.pretrained_model_path).cuda()
    # pred, truth = test(model)
    pred,truth,psnr_mean,ssim_mean = test(model)
    # name = opt.method +'epoch_20.mat'
    # name = 'gap_net_ic.mat'
    # print(f'Save reconstructed HSIs as {name}.')
    # scio.savemat(name, {'pred': pred})
   

if __name__ == '__main__':
    main()