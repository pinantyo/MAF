import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import _calculate_fan_in_and_fan_out
from timm.models.layers import to_2tuple, trunc_normal_


"""
MTL
"""
class Interpolate(nn.Module):

    def __init__(self, in_channels, out_channels, upsample=False):

        super().__init__()
        self.upsample = upsample
        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels, 
                out_channels, 
                (3, 3),
                stride=1, 
                padding=1, 
                bias=False
            )
        )

    def forward(self, x):
        x = self.block(x)
        if self.upsample:
            x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
        return 
    

class OverlapPatchUnembed(nn.Module):
    def __init__(self, in_c=48, out_c=3, bias=False, kernel_size=3):
        super(OverlapPatchUnembed, self).__init__()

        self.proj = nn.Conv2d(
            in_c, 
            out_c, 
            kernel_size=kernel_size, 
            stride=1, 
            padding=kernel_size // 2, 
            padding_mode='reflect', 
            bias=bias
        )

    def forward(self, x):
        x = self.proj(x)
        return x
    

"""
FROM RSHazeNet
"""
class LayerNormFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, bias, eps):
        ctx.eps = eps
        N, C, H, W = x.size()
        mu = x.mean(1, keepdim=True)
        var = (x - mu).pow(2).mean(1, keepdim=True)
        y = (x - mu) / (var + eps).sqrt()
        ctx.save_for_backward(y, var, weight)
        y = weight.view(1, C, 1, 1) * y + bias.view(1, C, 1, 1)
        return y

    @staticmethod
    def backward(ctx, grad_output):
        eps = ctx.eps

        N, C, H, W = grad_output.size()
        y, var, weight = ctx.saved_variables
        g = grad_output * weight.view(1, C, 1, 1)
        mean_g = g.mean(dim=1, keepdim=True)

        mean_gy = (g * y).mean(dim=1, keepdim=True)
        gx = 1. / torch.sqrt(var + eps) * (g - y * mean_gy - mean_g)
        return gx, (grad_output * y).sum(dim=3).sum(dim=2).sum(dim=0), grad_output.sum(dim=3).sum(dim=2).sum(
            dim=0), None

class LayerNorm2d(nn.Module):
    def __init__(self, channels, eps=1e-6):
        super(LayerNorm2d, self).__init__()
        self.register_parameter('weight', nn.Parameter(torch.ones(channels)))
        self.register_parameter('bias', nn.Parameter(torch.zeros(channels)))
        self.eps = eps

    def forward(self, x):
        return LayerNormFunction.apply(x, self.weight, self.bias, self.eps)

class OverlapPatchEmbed(nn.Module):
    def __init__(self, in_c=3, embed_dim=48, bias=False, kernel_size=3):
        super(OverlapPatchEmbed, self).__init__()

        self.proj = nn.Conv2d(
            in_c, 
            embed_dim, 
            kernel_size=kernel_size, 
            stride=1, 
            padding=kernel_size // 2, 
            padding_mode='reflect', 
            bias=bias
        )

    def forward(self, x):
        x = self.proj(x)

        return x

class Downsample(nn.Module):
    def __init__(self, n_feat, kernel_size=3):
        super(Downsample, self).__init__()
        self.body = nn.Sequential(
            nn.Conv2d(
                n_feat, 
                n_feat // 2, 
                kernel_size=to_2tuple(kernel_size), 
                padding=kernel_size // 2, 
                padding_mode='reflect', 
                bias=False
            ),
            nn.PixelUnshuffle(2)
        )
        
        
    def forward(self, x):
        return self.body(x)


class Upsample(nn.Module):
    def __init__(self, n_feat, kernel_size=3):
        super(Upsample, self).__init__()
        
        self.body = nn.Sequential(
            nn.Conv2d(
                n_feat, 
                n_feat * 2, 
                kernel_size=to_2tuple(kernel_size), 
                padding=kernel_size // 2, 
                padding_mode='reflect', 
                bias=False
            ),
            nn.PixelShuffle(2)
        )

    def forward(self, x):
        x = self.body(x)
        return x


# Cross-stage Multi-scale Interaction (CMIM) module
class CMFI(nn.Module):
    def __init__(self, dim, bias=False):
        super(CMFI, self).__init__()
        self.alpha = nn.Parameter(torch.ones(1))
        self.norm_1 = LayerNorm2d(dim)
        self.norm_2 = LayerNorm2d(dim * 2)

        self.q_1 = nn.Sequential(
            Downsample(dim),
            nn.Conv2d(dim * 2, dim * 2, kernel_size=to_2tuple(1), bias=bias)
        )

        self.v_1 = nn.Conv2d(dim, dim, kernel_size=to_2tuple(1), bias=bias)
        self.k_2 = nn.Conv2d(dim * 2, dim, kernel_size=to_2tuple(1), bias=bias)
        self.v_2 = nn.Conv2d(dim * 2, dim * 2, kernel_size=to_2tuple(1), bias=bias)

        self.proj_1 = nn.Conv2d(dim * 2, dim, kernel_size=to_2tuple(1), bias=bias)
        self.proj_2 = nn.Conv2d(dim, dim * 2, kernel_size=to_2tuple(1), bias=bias)

    def forward(self, x_1, x_2):
        input_1, input_2 = x_1, x_2
        
        x_1 = self.norm_1(x_1)
        x_2 = self.norm_2(x_2)

        b, c, h, w = x_2.shape
        q_1 = self.q_1(x_1).reshape(b, c, h * w)
        k_2 = self.k_2(x_2).reshape(b, c // 2, h * w)

        q_1 = F.normalize(q_1, dim=-1)
        k_2 = F.normalize(k_2, dim=-1)

        v_1 = self.v_1(x_1).reshape(b, c // 2, (h * 2) * (w * 2))
        v_2 = self.v_2(x_2).reshape(b, c, h * w)

        attn = (q_1 @ k_2.transpose(-2, -1)) * self.alpha

        attn_1 = attn.softmax(dim=-1)
        attn_2 = attn.transpose(-1, -2).softmax(dim=-1)

        x_1 = (attn_1 @ v_1).reshape(b, c, h * 2, w * 2)
        x_2 = (attn_2 @ v_2).reshape(b, c // 2, h, w)

        x_1 = self.proj_1(x_1) + input_1
        x_2 = self.proj_2(x_2) + input_2

        return x_1, x_2

# Intra-stage Transposed Fusion (ITFM) module
class IFTE(nn.Module):
    def __init__(self, dim, bias=False):
        super(IFTE, self).__init__()
        self.alpha = nn.Parameter(torch.ones(1))

        self.norm_dec = LayerNorm2d(dim)
        self.norm_skip = LayerNorm2d(dim)

        self.qk_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.qk = nn.Conv2d(dim * 2, dim * 2, kernel_size=to_2tuple(1), bias=False)

        self.v = nn.Conv2d(dim * 2, dim, kernel_size=to_2tuple(1), bias=bias)

        self.proj_out = nn.Conv2d(dim, dim, kernel_size=to_2tuple(1), bias=bias)

    def forward(self, x):
        x_dec, x_skip = x
        x_dec = self.norm_dec(x_dec)
        x_skip = self.norm_skip(x_skip)

        b, c, h, w = x[0].shape
        x = torch.cat((x_dec, x_skip), dim=1)

        q, k = self.qk(self.qk_avg_pool(x)).chunk(2, dim=1)
        v = self.v(x)
        q = q.reshape(b, c, 1)
        k = k.reshape(b, c, 1)
        v = v.reshape(b, c, h * w)

        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.alpha
        attn = attn.softmax(dim=-1)
        x = (attn @ v).reshape(b, c, h, w)

        x = self.proj_out(x)

        return x
    
"""
FROM gUNET
"""
class SKFusion(nn.Module):
    def __init__(self, dim, height=2, reduction=8):
        super(SKFusion, self).__init__()
        
        self.height = height
        d = max(int(dim/reduction), 4)
        
        self.mlp = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim, d, 1, bias=False),
            nn.ReLU(True),
            nn.Conv2d(d, dim*height, 1, bias=False)
        )
        self.softmax = nn.Softmax(dim=1)
        
        
    def forward(self, in_feats):
        B, C, H, W = in_feats[0].shape
        
        in_feats = torch.cat(in_feats, dim=1)
        in_feats = in_feats.view(B, self.height, C, H, W)
        feats_sum = torch.sum(in_feats, dim=1)
        attn = self.mlp(feats_sum)
        attn = self.softmax(attn.view(B, self.height, C, 1, 1))
        
        out = torch.sum(in_feats*attn, dim=1)
        return out
    
"""
INSPIRED FROM Res2Net
"""

class Res2NetBottleneck(nn.Module):
    def __init__(self, net_depth, inplanes, planes, kernel_size=3, scales=4, expansion=1):
        super(Res2NetBottleneck, self).__init__()
        if planes % scales != 0:
            raise ValueError('Planes must be divisible by scales')
        
        self.net_depth = net_depth
        self.expansion = expansion
        bottleneck_planes = planes
        
        self.norm = nn.BatchNorm2d(inplanes) # LayerNorm2d
        
        self.conv1_gated = nn.Sequential(
            nn.Conv2d(inplanes, bottleneck_planes, kernel_size=1),
            nn.Sigmoid()
        )
        
        self.conv1 = nn.Conv2d(inplanes, bottleneck_planes, kernel_size=1)
        self.conv2 = nn.ModuleList([
            nn.Conv2d(
                bottleneck_planes // scales, 
                bottleneck_planes // scales, 
                kernel_size=kernel_size, 
                stride=1, 
                padding=kernel_size // 2, 
                groups=bottleneck_planes // scales,
                padding_mode='reflect'
            )
            for _ in range(scales)
        ])

        self.proj_expansion = nn.Conv2d(
            bottleneck_planes, 
            planes * self.expansion, 
            kernel_size=1,
        )
                
        self.scales = scales
            
        # Init weights
        self.apply(self._init_weights)
        
    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            gain = (8 * self.net_depth) ** (-1/4)    # self.net_depth ** (-1/2), the deviation seems to be too small, a bigger one may be better
            fan_in, fan_out = _calculate_fan_in_and_fan_out(m.weight)
            std = gain * math.sqrt(2.0 / float(fan_in + fan_out))
            trunc_normal_(m.weight, std=std)
            
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
                
    def forward(self, x):
        out = self.norm(x)
        
        out_v = self.conv1_gated(out)
        out = self.conv1(out)
        
        xs = torch.chunk(out, self.scales, 1)
        ys = []
        for s in range(self.scales):
            if s == 0:
                ys.append(self.conv2[s-1](xs[s]))
            else:
                ys.append(self.conv2[s-1](xs[s] + ys[-1]))
                
        out = torch.cat(ys, 1)
        out = self.proj_expansion(out * out_v)
        
        return out

class ConvBlock(nn.Module):
    def __init__(self, net_depth, dim, kernel_size=3):
        super(ConvBlock, self).__init__()
        
        self.conv_layer = Res2NetBottleneck(
            net_depth=net_depth,
            inplanes=dim,
            planes=dim,
            scales=4,
            kernel_size=kernel_size
        )
        
        self.layer_scale = nn.Parameter(torch.ones((dim)), requires_grad=True)
        
    def forward(self, x):
        input_ = x

        # Local context
        x = self.conv_layer(x)
        
        # Scale
        x = self.layer_scale.unsqueeze(-1).unsqueeze(-1) * x
        
        # Convolution
        return x + input_


class DecoderBlock(nn.Module):
    def __init__(self, net_depths, depth, dim, upsample_iterate=1):
        super(DecoderBlock, self).__init__()
        self.upsample_iterate = upsample_iterate
        
        if self.upsample_iterate == 0:
            self.skip_connection_level = nn.Sequential(*[ConvBlock(net_depths, dim) for i in range(depth)])
        else:
            self.up = nn.ModuleList([Upsample(int(dim // 2**(i))) for i in range(upsample_iterate)])
            self.skip_connection_level = nn.ModuleList([nn.Sequential(
                *[ConvBlock(net_depths, int(dim // 2 ** (i + 1))) for _ in range(depth)]
            ) for i in range(upsample_iterate)])
            
    def forward(self, x):
        for i in range(self.upsample_iterate):
            x = self.up[i](x)
            x = self.skip_connection_level[i](x)
        
        if self.upsample_iterate == 0:
            x = self.skip_connection_level(x)
                                                         
        return x