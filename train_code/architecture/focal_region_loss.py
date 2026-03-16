import torch
import torch.nn as nn

import torch
import torch.nn as nn

import scipy.io as scio
def window_partition(x, window_size):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size

    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, C, H, W = x.shape
    x = x.view(B, C,H // window_size, window_size, W // window_size, window_size)
    windows = x.permute(0, 2, 4, 3, 5,1).contiguous().view(B, -1, window_size, window_size,C)
    return windows

class FocalRegionLoss(nn.Module):
    """The torch.nn.Module class that implements focal frequency loss - a
    frequency domain loss function for optimizing generative models.

    Ref:
    Focal Frequency Loss for Image Reconstruction and Synthesis. In ICCV 2021.
    <https://arxiv.org/pdf/2012.12821.pdf>

    Args:
        loss_weight (float): weight for focal frequency loss. Default: 1.0
        alpha (float): the scaling factor alpha of the spectrum weight matrix for flexibility. Default: 1.0
        patch_factor (int): the factor to crop image patches for patch-based focal frequency loss. Default: 1
        ave_spectrum (bool): whether to use minibatch average spectrum. Default: False
        log_matrix (bool): whether to adjust the spectrum weight matrix by logarithm. Default: False
        batch_matrix (bool): whether to calculate the spectrum weight matrix using batch-based statistics. Default: False
    """

    def __init__(self, loss_weight=1.0, alpha=1.0, log_matrix=True):
        super(FocalRegionLoss, self).__init__()
        self.loss_weight = loss_weight
        self.alpha = alpha
        # self.patch_factor = patch_factor
        # self.ave_spectrum = ave_spectrum
        self.log_matrix = log_matrix
        # self.batch_matrix = batch_matrix

    def loss_formulation(self, recon, real):
    
        recon = window_partition(recon,256)
        real  = window_partition(real,256)
        # if the matrix is calculated online: continuous, dynamic, based on current Euclidean distance
        matrix_tmp = torch.sqrt(torch.mean((recon - real) ** 2,dim=[-3,-2,-1],keepdim=True))
        # matrix_tmp_sum = torch.sum(matrix_tmp)
        # matrix_tmp = matrix_tmp/matrix_tmp_sum
        matrix_tmp = (1-matrix_tmp) ** self.alpha
        # matrix_tmp = (matrix_tmp) ** self.alpha
        if self.log_matrix:
            matrix_tmp = torch.log(matrix_tmp+1)
        matrix_tmp[torch.isnan(matrix_tmp)] = 0.0
        matrix_tmp = torch.clamp(matrix_tmp, min=0.0, max=1.0)
       
        # weight_matrix = matrix_tmp.clone().detach()
        weight_matrix = matrix_tmp.clone().detach()
        assert weight_matrix.min().item() >= 0 and weight_matrix.max().item() <= 1, (
            'The values of spectrum weight matrix should be in the range [0, 1], '
            'but got Min: %.10f Max: %.10f' % (weight_matrix.min().item(), weight_matrix.max().item()))

        # frequency distance using (squared) Euclidean distance
        
        pixel_loss = torch.sqrt(torch.mean((recon - real) ** 2,dim=[-3,-2,-1],keepdim=True))

        # dynamic spectrum weighting (Hadamard product)
        loss = weight_matrix * pixel_loss
        return torch.mean(loss)

    def forward(self, pred, target, **kwargs):
        """Forward function to calculate focal frequency loss.

        Args:
            pred (torch.Tensor): of shape (N, C, H, W). Predicted tensor.
            target (torch.Tensor): of shape (N, C, H, W). Target tensor.
            matrix (torch.Tensor, optional): Element-wise spectrum weight matrix.
                Default: None (If set to None: calculated online, dynamic).
        """
        # calculate focal frequency loss
        return self.loss_formulation(pred, target) * self.loss_weight
'''
def window_partition(x, window_size):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size

    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, C, H, W = x.shape
    x = x.view(B, C,H // window_size, window_size, W // window_size, window_size)
    windows = x.permute(0, 1, 2, 4, 3, 5).contiguous().view(B, C,-1, window_size, window_size)
    return windows

class FocalRegionLoss(nn.Module):
    """The torch.nn.Module class that implements focal frequency loss - a
    frequency domain loss function for optimizing generative models.

    Ref:
    Focal Frequency Loss for Image Reconstruction and Synthesis. In ICCV 2021.
    <https://arxiv.org/pdf/2012.12821.pdf>

    Args:
        loss_weight (float): weight for focal frequency loss. Default: 1.0
        alpha (float): the scaling factor alpha of the spectrum weight matrix for flexibility. Default: 1.0
        patch_factor (int): the factor to crop image patches for patch-based focal frequency loss. Default: 1
        ave_spectrum (bool): whether to use minibatch average spectrum. Default: False
        log_matrix (bool): whether to adjust the spectrum weight matrix by logarithm. Default: False
        batch_matrix (bool): whether to calculate the spectrum weight matrix using batch-based statistics. Default: False
    """

    def __init__(self, loss_weight=1.0, alpha=1.0, log_matrix=True):
        super(FocalRegionLoss, self).__init__()
        self.loss_weight = loss_weight
        self.alpha = alpha
        # self.patch_factor = patch_factor
        # self.ave_spectrum = ave_spectrum
        self.log_matrix = log_matrix
        # self.batch_matrix = batch_matrix

    def loss_formulation(self, recon, real):
    
        recon = window_partition(recon,8)
        real  = window_partition(real,8)
        # if the matrix is calculated online: continuous, dynamic, based on current Euclidean distance
        matrix_tmp = torch.sqrt(torch.mean((recon - real) ** 2,dim=[-2,-1],keepdim=True))
        # matrix_tmp_sum = torch.sum(matrix_tmp)
        # matrix_tmp = matrix_tmp/matrix_tmp_sum
        matrix_tmp = (1-matrix_tmp) ** self.alpha
        # matrix_tmp = (matrix_tmp) ** self.alpha
        if self.log_matrix:
            matrix_tmp = torch.log(matrix_tmp+1)
        matrix_tmp[torch.isnan(matrix_tmp)] = 0.0
        matrix_tmp = torch.clamp(matrix_tmp, min=0.0, max=1.0)
       
        # weight_matrix = matrix_tmp.clone().detach()
        weight_matrix = matrix_tmp.clone().detach()
        assert weight_matrix.min().item() >= 0 and weight_matrix.max().item() <= 1, (
            'The values of spectrum weight matrix should be in the range [0, 1], '
            'but got Min: %.10f Max: %.10f' % (weight_matrix.min().item(), weight_matrix.max().item()))

        # frequency distance using (squared) Euclidean distance
        
        pixel_loss = torch.sqrt(torch.mean((recon - real) ** 2,dim=[-2,-1],keepdim=True))

        # dynamic spectrum weighting (Hadamard product)
        loss = weight_matrix * pixel_loss
        return torch.mean(loss)

    def forward(self, pred, target, **kwargs):
        """Forward function to calculate focal frequency loss.

        Args:
            pred (torch.Tensor): of shape (N, C, H, W). Predicted tensor.
            target (torch.Tensor): of shape (N, C, H, W). Target tensor.
            matrix (torch.Tensor, optional): Element-wise spectrum weight matrix.
                Default: None (If set to None: calculated online, dynamic).
        """
        # calculate focal frequency loss
        return self.loss_formulation(pred, target) * self.loss_weight
'''



class FocalSpectralLoss(nn.Module):
    """The torch.nn.Module class that implements focal frequency loss - a
    frequency domain loss function for optimizing generative models.

    Ref:
    Focal Frequency Loss for Image Reconstruction and Synthesis. In ICCV 2021.
    <https://arxiv.org/pdf/2012.12821.pdf>

    Args:
        loss_weight (float): weight for focal frequency loss. Default: 1.0
        alpha (float): the scaling factor alpha of the spectrum weight matrix for flexibility. Default: 1.0
        patch_factor (int): the factor to crop image patches for patch-based focal frequency loss. Default: 1
        ave_spectrum (bool): whether to use minibatch average spectrum. Default: False
        log_matrix (bool): whether to adjust the spectrum weight matrix by logarithm. Default: False
        batch_matrix (bool): whether to calculate the spectrum weight matrix using batch-based statistics. Default: False
    """

    def __init__(self, loss_weight=1.0, alpha=1.0, log_matrix=True):
        super(FocalSpectralLoss, self).__init__()
        self.loss_weight = loss_weight
        self.alpha = alpha
        # self.patch_factor = patch_factor
        # self.ave_spectrum = ave_spectrum
        self.log_matrix = log_matrix
        # self.batch_matrix = batch_matrix

    def loss_formulation(self, recon, real):
    
    
        # if the matrix is calculated online: continuous, dynamic, based on current Euclidean distance
        matrix_tmp = torch.sqrt(torch.mean((recon - real) ** 2,dim=[2,3],keepdim=True))
        # matrix_tmp_sum = torch.sum(matrix_tmp)
        # matrix_tmp = matrix_tmp/matrix_tmp_sum
        matrix_tmp = (1-matrix_tmp) ** self.alpha
        if self.log_matrix:
            matrix_tmp = torch.log(matrix_tmp+1)
        matrix_tmp[torch.isnan(matrix_tmp)] = 0.0
        matrix_tmp = torch.clamp(matrix_tmp, min=0.0, max=1.0)
       
        # weight_matrix = matrix_tmp.clone().detach()
        weight_matrix = matrix_tmp.clone().detach()
        assert weight_matrix.min().item() >= 0 and weight_matrix.max().item() <= 1, (
            'The values of spectrum weight matrix should be in the range [0, 1], '
            'but got Min: %.10f Max: %.10f' % (weight_matrix.min().item(), weight_matrix.max().item()))

        # frequency distance using (squared) Euclidean distance
        
        pixel_loss = torch.sqrt(torch.mean((recon - real) ** 2,dim=[2,3],keepdim=True))

        # dynamic spectrum weighting (Hadamard product)
        loss = weight_matrix * pixel_loss
        return torch.sum(loss)

    def forward(self, pred, target, **kwargs):
        """Forward function to calculate focal frequency loss.

        Args:
            pred (torch.Tensor): of shape (N, C, H, W). Predicted tensor.
            target (torch.Tensor): of shape (N, C, H, W). Target tensor.
            matrix (torch.Tensor, optional): Element-wise spectrum weight matrix.
                Default: None (If set to None: calculated online, dynamic).
        """
        # calculate focal frequency loss
        return self.loss_formulation(pred, target) * self.loss_weight
    
if __name__=='__main__':
    loss_model = FocalRegionLoss(loss_weight=1,alpha=1,log_matrix=True)
    pred = scio.loadmat('/data/users/zhaowangcai01/MST-main/ours_epoch300.mat')['pred']
    res = torch.rand(1,28,256,256)
    gt = torch.rand(1,28,256,256)
    loss = loss_model(res,gt)
    print(loss.shape)

    
