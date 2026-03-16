import argparse
import template

parser = argparse.ArgumentParser(description="HyperSpectral Image Reconstruction Toolbox")
parser.add_argument('--template', default='tsa_net',
                    help='You can set various templates in option.py')

# Hardware specifications
parser.add_argument("--gpu_id", type=str, default='4,5')

# Data specifications
parser.add_argument('--data_root', type=str, default='../../datasets/', help='dataset directory')
parser.add_argument("--max_epoch", type=int, default=300, help='total epoch')
# Saving specifications
parser.add_argument('--outf', type=str, default='./exp/mst_l/', help='saving_path')

# Model specifications
parser.add_argument('--method', type=str, default='tsa_net', help='method name')
parser.add_argument('--pretrained_model_path', type=str, default='/data/users/zhaowangcai01/MST-main/simulation/train_code/exp/tsanet/2023_10_04_05_25_17/model/model_epoch_300.pth', help='pretrained model directory')
parser.add_argument("--input_setting", type=str, default='H',
                    help='the input measurement of the network: H, HM or Y')
parser.add_argument("--input_mask", type=str, default='Phi',
                    help='the input mask of the network: Phi, Phi_PhiPhiT or None')

opt = parser.parse_args()
template.set_template(opt)

opt.mask_path = f"{opt.data_root}/TSA_simu_data/"
opt.test_path = f"{opt.data_root}/TSA_simu_data/Truth/"

for arg in vars(opt):
    if vars(opt)[arg] == 'True':
        vars(opt)[arg] = True
    elif vars(opt)[arg] == 'False':
        vars(opt)[arg] = False