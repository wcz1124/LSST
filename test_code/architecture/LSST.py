import torch.nn as nn
import torch
import torch.nn.functional as F
from einops import rearrange
import math
import warnings
from torch.nn.init import _calculate_fan_in_and_fan_out

def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    def norm_cdf(x):
        return (1. + math.erf(x / math.sqrt(2.))) / 2.

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn("mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
                      "The distribution of values may be incorrect.",
                      stacklevel=2)
    with torch.no_grad():
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)
        tensor.uniform_(2 * l - 1, 2 * u - 1)
        tensor.erfinv_()
        tensor.mul_(std * math.sqrt(2.))
        tensor.add_(mean)
        tensor.clamp_(min=a, max=b)
        return tensor


def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    # type: (Tensor, float, float, float, float) -> Tensor
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)


def variance_scaling_(tensor, scale=1.0, mode='fan_in', distribution='normal'):
    fan_in, fan_out = _calculate_fan_in_and_fan_out(tensor)
    if mode == 'fan_in':
        denom = fan_in
    elif mode == 'fan_out':
        denom = fan_out
    elif mode == 'fan_avg':
        denom = (fan_in + fan_out) / 2
    variance = scale / denom
    if distribution == "truncated_normal":
        trunc_normal_(tensor, std=math.sqrt(variance) / .87962566103423978)
    elif distribution == "normal":
        tensor.normal_(std=math.sqrt(variance))
    elif distribution == "uniform":
        bound = math.sqrt(3 * variance)
        tensor.uniform_(-bound, bound)
    else:
        raise ValueError(f"invalid distribution {distribution}")


def lecun_normal_(tensor):
    variance_scaling_(tensor, mode='fan_in', distribution='truncated_normal')

class Gated_Dconv_FeedForward(nn.Module):
    def __init__(self, 
                 dim, 
                 ffn_expansion_factor = 2.66
    ):
        super(Gated_Dconv_FeedForward, self).__init__()

        hidden_features = int(dim*ffn_expansion_factor)

        self.project_in = nn.Conv2d(dim, hidden_features*2, kernel_size=1, bias=False)

        self.dwconv = nn.Conv2d(hidden_features*2, hidden_features*2, kernel_size=3, stride=1, padding=1, groups=hidden_features*2, bias=True)

        self.act_fn = nn.GELU()

        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=False)

    def forward(self, x):
        """
        x: [b, c, h, w]
        return out: [b, c, h, w]
        """
        x = x.permute(0, 3, 1, 2)
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = self.act_fn(x1) * x2
        x = self.project_out(x)
        x = x.permute(0, 2, 3, 1)
        return x

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)

    def forward(self, x, *args, **kwargs):
        x = self.norm(x)
        return self.fn(x, *args, **kwargs)


class GELU(nn.Module):
    def forward(self, x):
        return F.gelu(x)

def conv(in_channels, out_channels, kernel_size, bias=False, padding = 1, stride = 1):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size//2), bias=bias, stride=stride)

def channel_shuffle(x, groups):
    batchsize, height,weight, num_channels = x.size()
    channels_per_group = num_channels // groups

    # reshape
    x = x.view(batchsize, height,weight, groups,
               channels_per_group)

    x = torch.transpose(x, -1, -2).contiguous()

    # flatten
    x = x.view(batchsize,height,weight,-1)

    return x

def channel_shuffle_back(x, groups):
    batchsize, height,weight, num_channels = x.size()
    channels_per_group = num_channels // groups

    # reshape
    x = x.view(batchsize, height,weight,
               channels_per_group, groups)

    x = torch.transpose(x, -1, -2).contiguous()

    # flatten
    x = x.view(batchsize,height,weight,-1)

    return x

def shift_back(inputs,step=2):          # input [bs,28,256,310]  output [bs, 28, 256, 256]
    [bs, nC, row, col] = inputs.shape
    down_sample = 256//row
    step = float(step)/float(down_sample*down_sample)
    out_col = row
    for i in range(nC):
        inputs[:,i,:,:out_col] = \
            inputs[:,i,:,int(step*i):int(step*i)+out_col]
    return inputs[:, :, :, :out_col]


class ESS_MSEB(nn.Module):
    def __init__(
            self,
            dim,
            dim_head=64,
            heads=8,
            groups = 4,
    ):
        super().__init__()
        
        self.groups = groups
        self.num_heads = heads
        self.dim_head = dim_head
        self.dim = dim
        self.atten = MS_MSA(dim=self.dim//self.groups, dim_head=self.dim_head//self.groups, heads=self.num_heads)

    def forward(self, x_in):
        """
        x_in: [b,h,w,c]
        mask: [1,h,w,c]
        return out: [b,h,w,c]
        """
        b, h, w, c = x_in.shape
        x = x_in
        channel_per_group = c//self.groups
        x0 = x[:,:,:,0:channel_per_group]
        x1 = x[:,:,:,channel_per_group:channel_per_group*2]
        x2 = x[:,:,:,channel_per_group*2:channel_per_group*3]
        x3 = x[:,:,:,channel_per_group*3:]
        x0_out = self.atten(x0)
        x1_out = self.atten(x1)
        x2_out = self.atten(x2)
        x3_out = self.atten(x3)
        out = torch.cat((x0_out,x1_out,x2_out,x3_out),dim=-1)
        return out


class ESS_MSEG(nn.Module):
    def __init__(
            self,
            dim,
            dim_head=64,
            heads=8,
            groups = 4,
    ):
        super().__init__()
        
        self.groups = groups
        self.num_heads = heads
        self.dim_head = dim_head
        self.dim = dim
        self.atten = MS_MSA(dim=self.dim//self.groups, dim_head=self.dim_head//self.groups, heads=self.num_heads)

    def forward(self, x_in):
        """
        x_in: [b,h,w,c]
        mask: [1,h,w,c]
        return out: [b,h,w,c]
        """
        b, h, w, c = x_in.shape
        # x = x_in.reshape(b,h*w,c)

        x = channel_shuffle(x_in,groups=self.groups)
        channel_per_group = c//self.groups
        x0 = x[:,:,:,0:channel_per_group]
        x1 = x[:,:,:,channel_per_group:channel_per_group*2]
        x2 = x[:,:,:,channel_per_group*2:channel_per_group*3]
        x3 = x[:,:,:,channel_per_group*3:]
        x0_out = self.atten(x0)
        x1_out = self.atten(x1)
        x2_out = self.atten(x2)
        x3_out = self.atten(x3)
        out = torch.cat((x0_out,x1_out,x2_out,x3_out),dim=-1)
        out = channel_shuffle_back(out,groups=self.groups)
        return out

def conv_1x1_bn(inp, oup):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        nn.SiLU()
    )

def conv_1x1_gelu(inp, oup):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        GELU()
    )
def conv_nxn_bn(inp, oup, kernal_size=3, stride=1):
    return nn.Sequential(
        nn.Conv2d(inp, oup, kernal_size, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        nn.SiLU()
    )

def dconv_nxn_bn(inp, kernal_size=3, stride=1):
    return nn.Sequential(
        nn.Conv2d(inp, inp, kernal_size, stride, 1,groups=inp, bias=False),
        nn.BatchNorm2d(inp),
        nn.SiLU()
    )

class ESS_BL(nn.Module):
    def __init__(
            self,
            dim,
            dim_head=64,
            heads=8,
            num_blocks=2,
            groups = 4
    ):
        super().__init__()
        # Local spatial representations
        # self.conv1 = dconv_nxn_bn(dim,kernal_size=3,stride=1)
        # self.conv2 = conv_1x1_bn(dim, dim)
        self.lcb = CBlock(dim=dim)
        # fusion
        self.fusion = conv_1x1_gelu(2*dim, dim)
        
        self.blocks = nn.ModuleList([])
        for _ in range(num_blocks):
            self.blocks.append(nn.ModuleList([
                ESS_MSEB(dim=dim, dim_head=dim_head, heads=heads,groups=groups),
                # PreNorm(dim, Gated_Dconv_FeedForward(dim=dim)),
                ESS_MSEG(dim=dim, dim_head=dim_head, heads=heads,groups=groups),
                PreNorm(dim, Gated_Dconv_FeedForward(dim=dim))
            ]))

    def forward(self, x,mask=None):
        """
        x: [b,c,h,w]
        return out: [b,c,h,w]
        """
        res = x.clone()

        # Local representations
        # x = self.conv1(x)
        # x = self.conv2(x)
        # lcr = x.clone()
        
        x = x.permute(0, 2, 3, 1)
        lcr = self.lcb(x)
        for (attn1,attn2,ff2) in self.blocks:
            x = attn1(x) + x
            # x = ff1(x) + x
            x = attn2(x) + x
            x = ff2(x) + x
        out_atten = x.permute(0, 3, 1, 2)
        fs = self.fusion(
                torch.cat((lcr, out_atten), dim=1)
            )
        out = fs + res
        return out




class LSST(nn.Module):
    def __init__(self, in_dim=28, out_dim=28, dim=28, stage=2, num_blocks=[2,4,4]):
        super(LSST, self).__init__()
        self.dim = dim
        self.stage = stage
        self.fution = nn.Conv2d(56, 28, 1, padding=0, bias=True)
        # Input projection
        self.embedding = nn.Conv2d(in_dim, self.dim, 3, 1, 1, bias=False)

        # Encoder
        self.encoder_layers = nn.ModuleList([])
        dim_stage = dim
        for i in range(stage):
            self.encoder_layers.append(nn.ModuleList([
                ESS_BL(
                    dim=dim_stage, num_blocks=num_blocks[i], dim_head=dim, heads=dim_stage // dim),
                nn.Conv2d(dim_stage, dim_stage * 2, 4, 2, 1, bias=False),
            ]))
            dim_stage *= 2

        # Bottleneck
        self.bottleneck = ESS_BL(
            dim=dim_stage, dim_head=dim, heads=dim_stage // dim, num_blocks=num_blocks[-1])

        # Decoder
        self.decoder_layers = nn.ModuleList([])
        for i in range(stage):
            self.decoder_layers.append(nn.ModuleList([
                nn.ConvTranspose2d(dim_stage, dim_stage // 2, stride=2, kernel_size=2, padding=0, output_padding=0),
                nn.Conv2d(dim_stage, dim_stage // 2, 1, 1, bias=False),
                ESS_BL(
                    dim=dim_stage // 2, num_blocks=num_blocks[stage - 1 - i], dim_head=dim,
                    heads=(dim_stage // 2) // dim),
            ]))
            dim_stage //= 2

        # Output projection
        self.mapping = nn.Conv2d(self.dim, out_dim, 3, 1, 1, bias=False)

        #### activation function
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
    

    def initial_x(self, x, Phi):
        """
        :param y: [b,28,256,310]
        :param Phi: [b,28,256,256]
        :return: z: [b,28,256,256]
        """
        x = self.fution(torch.cat([x, Phi], dim=1))
        return x
    
    def forward(self, x,Phi=None):
        """
        x: [b,c,h,w]
        return out:[b,c,h,w]
        """
        if Phi==None:
            Phi = torch.rand((1,28,256,256)).cuda()
        x = self.initial_x(x, Phi)
        # Embedding
        fea = self.embedding(x)

        # Encoder
        fea_encoder = []
        for (ESS_BL, FeaDownSample) in self.encoder_layers:
            fea = ESS_BL(fea)
            fea_encoder.append(fea)
            fea = FeaDownSample(fea)

        # Bottleneck
        fea = self.bottleneck(fea)

        # Decoder
        for i, (FeaUpSample, Fution, LeWinBlcok) in enumerate(self.decoder_layers):
            fea = FeaUpSample(fea)
            fea = Fution(torch.cat([fea, fea_encoder[self.stage-1-i]], dim=1))
            fea = LeWinBlcok(fea)

        # Mapping
        out = self.mapping(fea) + x

        return out

class MS_MSA(nn.Module):
    def __init__(
            self,
            dim,
            dim_head,
            heads,
    ):
        super().__init__()
        self.num_heads = heads
        self.dim_head = dim_head
        self.to_q = nn.Linear(dim, dim_head * heads, bias=False)
        self.to_k = nn.Linear(dim, dim_head * heads, bias=False)
        self.to_v = nn.Linear(dim, dim_head * heads, bias=False)
        self.rescale = nn.Parameter(torch.ones(heads, 1, 1))
        self.proj = nn.Linear(dim_head * heads, dim, bias=True)
        # self.pos_emb = nn.Sequential(
        #     nn.Conv2d(dim, dim, 3, 1, 1, bias=False, groups=dim),
        #     GELU(),
        #     nn.Conv2d(dim, dim, 3, 1, 1, bias=False, groups=dim),
        # )
        self.dim = dim

    def forward(self, x_in):
        """
        x_in: [b,h,w,c]
        return out: [b,h,w,c]
        """
        b, h, w, c = x_in.shape
        x = x_in.reshape(b,h*w,c)
        q_inp = self.to_q(x)
        k_inp = self.to_k(x)
        v_inp = self.to_v(x)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.num_heads),
                                (q_inp, k_inp, v_inp))
        v = v
        # q: b,heads,hw,c
        q = q.transpose(-2, -1)
        k = k.transpose(-2, -1)
        v = v.transpose(-2, -1)
        q = F.normalize(q, dim=-1, p=2)
        k = F.normalize(k, dim=-1, p=2)
        attn = (k @ q.transpose(-2, -1))   # A = K^T*Q
        attn = attn * self.rescale
        attn = attn.softmax(dim=-1)
        x = attn @ v   # b,heads,d,hw
        x = x.permute(0, 3, 1, 2)    # Transpose
        x = x.reshape(b, h * w, self.num_heads * self.dim_head)
        out_c = self.proj(x).view(b, h, w, c)
        # out_p = self.pos_emb(v_inp.reshape(b,h,w,c).permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        out = out_c

        return out
    
'''
class FeedForward(nn.Module):
    def __init__(self, dim, mult=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(dim, dim * mult, 1, 1, bias=False),
            GELU(),
            nn.Conv2d(dim * mult, dim * mult, 3, 1, 1, bias=False, groups=dim * mult),
            GELU(),
            nn.Conv2d(dim * mult, dim, 1, 1, bias=False),
        )

    def forward(self, x):
        """
        x: [b,h,w,c]
        return out: [b,h,w,c]
        """
        out = self.net(x.permute(0, 3, 1, 2))
        # return out.permute(0, 2, 3, 1)
        return out
'''
'''
class FeedForward(nn.Module):
    def __init__(self, dim, mult=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(dim, dim * mult, 1, 1, bias=False),
            GELU(),
            nn.Conv2d(dim * mult, dim * mult, 3, 1, 1, bias=False, groups=dim * mult),
            GELU(),
            nn.Conv2d(dim * mult, dim, 1, 1, bias=False),
        )
        self.mask = nn.Conv2d(dim , dim, 1, 1, bias=False)
        self.fu = nn.Conv2d(dim,dim, 1, 1, bias=False)
    def forward(self, x,mask):
        """
        x: [b,h,w,c]
        return out: [b,h,w,c]
        """
        res = x.clone()
        mask = self.mask(mask)

        out = self.net(x.permute(0, 3, 1, 2))
        out = out*mask
        out = self.fu(out)
        out = out + res.permute(0, 3, 1, 2)
        # return out.permute(0, 2, 3, 1)
        return out
'''
class FeedForward(nn.Module):
    def __init__(self, dim, mult=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(dim, dim * mult, 1, 1, bias=False),
            GELU(),
            nn.Conv2d(dim * mult, dim * mult, 3, 1, 1, bias=False, groups=dim * mult),
            GELU(),
            nn.Conv2d(dim * mult, dim, 1, 1, bias=False),
        )

    def forward(self, x):
        """
        x: [b,h,w,c]
        return out: [b,h,w,c]
        """
        res = x.clone()
        out = self.net(x.permute(0, 3, 1, 2))
        out = out + res.permute(0, 3, 1, 2)
        # return out.permute(0, 2, 3, 1)
        return out





class CBlock(nn.Module):
    r""" ConvNeXt Block. There are two equivalent implementations:
    (1) DwConv -> LayerNorm (channels_first) -> 1x1 Conv -> GELU -> 1x1 Conv; all in (N, C, H, W)
    (2) DwConv -> Permute to (N, H, W, C); LayerNorm (channels_last) -> Linear -> GELU -> Linear; Permute back
    We use (2) as we find it slightly faster in PyTorch
    
    Args:
        dim (int): Number of input channels.
        drop_path (float): Stochastic depth rate. Default: 0.0
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
    """
    def __init__(self, dim):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim) # depthwise conv
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim) # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        # self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim)), 
                                    # requires_grad=True) if layer_scale_init_value > 0 else None
       

    def forward(self, x):
        # (N, H, W,C) -> (N, C,H, W)
        x = x.permute(0, 3,1,2)
        input = x

        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1) # (N, C, H, W) -> (N, H, W, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        # if self.gamma is not None:
        #     x = self.gamma * x
        x = x.permute(0, 3, 1, 2) # (N, H, W, C) -> (N, C, H, W)

        x = input + x
        return x



