import torch
from .MST import MST
from .GAP_Net import GAP_net
from .ADMM_Net import ADMM_net
from .TSA_Net import TSA_Net
from .HDNet import HDNet, FDL
from .DGSMP import HSI_CS
from .BIRNAT import BIRNAT
from .MST_Plus_Plus import MST_Plus_Plus
from .Lambda_Net import Lambda_Net
from .CST import CST
from .DAUHST import DAUHST
from .padut import PADUT
# from .BiSRNet import BiSRNet
from .ESST import ESST
# from .MESST_Plus_Plus import MESST
from .ESST_NMM import ESST_NMM
from .MESST import MESST
from .Baseline import BMST
from .ESST_Plus_Plus import ESST_Plus_Plus
from .ESST_Plus_Plus_Hd import ESST_Plus_Plus_Hd
from .MESST_Plus_Plus import MESST_Plus_Plus
from .LSSTM_Plus_Plus import LSSTM_Plus_Plus
from .MESST_Plus_Plus import MESSTNM
from .LSST import LSST
from .LSSTM import LSSTM
from .BaseGSA import GMST
from .BaseWSA import WMST
def model_generator(method, pretrained_model_path=None):
    if method == 'mst_s':
        model = MST(dim=28, stage=2, num_blocks=[2, 2, 2]).cuda()
    elif method == 'mst_m':
        model = MST(dim=28, stage=2, num_blocks=[2, 4, 4]).cuda()
    elif method == 'mst_l':
        model = MST(dim=28, stage=2, num_blocks=[4, 7, 5]).cuda()
    elif method == 'esst':
        model = ESST(dim=28, stage=2, num_blocks=[1, 1, 1]).cuda()
    elif method == 'esst_m':
        model = ESST(dim=28, stage=2, num_blocks=[2, 2, 2]).cuda()
    elif method == 'esst_l':
        model = ESST(dim=28, stage=2, num_blocks=[2, 4, 4]).cuda()
    elif method == 'bmst':
        model = BMST(dim=28, stage=2, num_blocks=[1, 1, 2]).cuda()
    elif method == 'gmst':
        model = GMST(dim=28, stage=2, num_blocks=[1, 1, 2]).cuda()
    elif method == 'wmst':
        model = WMST(dim=28, stage=2, num_blocks=[1, 1, 2]).cuda()
    elif method == 'messt':
        model = MESST(dim=28, stage=2, num_blocks=[1, 1, 1]).cuda()
    elif method == 'messt_m':
        model = MESST(dim=28, stage=2, num_blocks=[2, 2, 2]).cuda()
    elif method == 'messt_l':
        model = MESST(dim=28, stage=2, num_blocks=[2, 4, 4]).cuda()
    elif method == 'esst_nmm':
        model = ESST_NMM(dim=28, stage=2, num_blocks=[1, 1, 1]).cuda()
    elif method == 'lsst_s':
        model = LSST(dim=28, stage=2, num_blocks=[2, 2, 2]).cuda()
        
    elif method == 'lsst_m':
        model = LSST(dim=28, stage=2, num_blocks=[4, 4, 4]).cuda()
        
    elif method == 'lsst_s':
        model = LSST(dim=28, stage=2, num_blocks=[2, 2, 2]).cuda()
        
    elif method == 'lsstm_s':
        model = LSSTM(dim=28, stage=2, num_blocks=[1, 1, 2]).cuda()
        
    elif method == 'lsstm_m':
        model = LSSTM(dim=28, stage=2, num_blocks=[2, 2, 2]).cuda()
        
    elif method == 'lsstm_l':
        model = LSSTM(dim=28, stage=2, num_blocks=[2, 3, 3]).cuda()
        
    elif method == 'gap_net':
        model = GAP_net().cuda()
    elif method == 'admm_net':
        model = ADMM_net().cuda()
    elif method == 'tsa_net':
        model = TSA_Net().cuda()
    elif method == 'hdnet':
        model = HDNet().cuda()
        fdl_loss = FDL(loss_weight=0.7,
             alpha=2.0,
             patch_factor=4,
             ave_spectrum=True,
             log_matrix=True,
             batch_matrix=True,
             ).cuda()
    elif method == 'dgsmp':
        model = HSI_CS(Ch=28, stages=4).cuda()
    elif method == 'birnat':
        model = BIRNAT().cuda()
    elif method == 'mst_plus_plus':
        model = MST_Plus_Plus(in_channels=28, out_channels=28, n_feat=28, stage=3).cuda()
    elif method == 'esst_plus_plus':
        model = ESST_Plus_Plus(in_channels=28, out_channels=28, n_feat=28, stage=3).cuda()
    elif method == 'esst_plus_plus_hd':
        model = ESST_Plus_Plus_Hd(in_channels=31, out_channels=31, n_feat=31, stage=3).cuda()
    elif method == 'messt_plus_plus':
        model = MESST_Plus_Plus(in_channels=28, out_channels=28, n_feat=28, stage=3).cuda()
    elif method == 'lsstm_plus_plus':
        model = LSSTM_Plus_Plus(in_channels=28, out_channels=28, n_feat=28, stage=3).cuda()
        
    elif method == 'messtnm':
        model = MESSTNM(dim=28, stage=2, num_blocks=[1, 1, 1]).cuda()
    elif method == 'lambda_net':
        model = Lambda_Net(out_ch=28).cuda()
    elif method == 'cst_s':
        model = CST(num_blocks=[1, 1, 2], sparse=True).cuda()
    elif method == 'cst_m':
        model = CST(num_blocks=[2, 2, 2], sparse=True).cuda()
    elif method == 'cst_l':
        model = CST(num_blocks=[2, 4, 6], sparse=True).cuda()
    elif method == 'cst_l_plus':
        model = CST(num_blocks=[2, 4, 6], sparse=False).cuda()
    elif 'dauhst' in method:
        num_iterations = int(method.split('_')[1][0])
        model = DAUHST(num_iterations=num_iterations).cuda()
    
   
    elif 'padut' in method:
        num_iterations = int(method.split('_')[-1])
        model = PADUT(in_c=28, n_feat=28,nums_stages=num_iterations-1).cuda()
   
    
    # elif method == 'bisrnet':
    #     model = BiSRNet(in_channels=28, out_channels=28, n_feat=28, stage=1, num_blocks=[1,1,1]).cuda()
    else:
        print(f'Method {method} is not defined !!!!')
    if pretrained_model_path is not None:
        print(f'load model from {pretrained_model_path}')
        checkpoint = torch.load(pretrained_model_path)
        model.load_state_dict({k.replace('module.', ''): v for k, v in checkpoint.items()},
                              strict=True)
    if method == 'hdnet':
        return model,fdl_loss
    return model