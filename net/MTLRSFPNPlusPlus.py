import torch
import torch.nn as nn
from net.common import OverlapPatchEmbed, OverlapPatchUnembed, Downsample, CMFI, SKFusion, ConvBlock, DecoderBlock


"""
    From A2-FPN
"""
def l2_norm(x):
    return torch.einsum("bcn, bn->bcn", x, 1 / torch.norm(x, p=2, dim=-2))


class ConvBnRelu(nn.Module):
    def __init__(self, in_planes, out_planes, ksize, stride, pad, dilation=1,
                 groups=1, has_bn=True, norm_layer=nn.BatchNorm2d, bn_eps=1e-5,
                 has_relu=True, inplace=True, has_bias=False):
        super(ConvBnRelu, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=ksize,
                              stride=stride, padding=pad,
                              dilation=dilation, groups=groups, bias=has_bias)
        self.has_bn = has_bn
        if self.has_bn:
            self.bn = norm_layer(out_planes, eps=bn_eps)
        self.has_relu = has_relu
        if self.has_relu:
            self.relu = nn.ReLU(inplace=inplace)

    def forward(self, x):
        x = self.conv(x)
        if self.has_bn:
            x = self.bn(x)
        if self.has_relu:
            x = self.relu(x)

        return x


class Attention(nn.Module):
    def __init__(self, in_places, scale=8, eps=1e-6):
        super(Attention, self).__init__()
        self.gamma = nn.Parameter(torch.zeros(1))
        self.in_places = in_places
        self.l2_norm = l2_norm
        self.eps = eps

        self.query_conv = nn.Conv2d(in_channels=in_places, out_channels=in_places // scale, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_places, out_channels=in_places // scale, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_places, out_channels=in_places, kernel_size=1)

    def forward(self, x):
        # Apply the feature map to the queries and keys
        batch_size, chnnels, width, height = x.shape
        Q = self.query_conv(x).view(batch_size, -1, width * height)
        K = self.key_conv(x).view(batch_size, -1, width * height)
        V = self.value_conv(x).view(batch_size, -1, width * height)

        Q = self.l2_norm(Q).permute(-3, -1, -2)
        K = self.l2_norm(K)

        tailor_sum = 1 / (width * height + torch.einsum("bnc, bc->bn", Q, torch.sum(K, dim=-1) + self.eps))
        value_sum = torch.einsum("bcn->bc", V).unsqueeze(-1)
        value_sum = value_sum.expand(-1, chnnels, width * height)

        matrix = torch.einsum('bmn, bcn->bmc', K, V)
        matrix_sum = value_sum + torch.einsum("bnm, bmc->bcn", Q, matrix)

        weight_value = torch.einsum("bcn, bn->bcn", matrix_sum, tailor_sum)
        weight_value = weight_value.view(batch_size, chnnels, height, width)

        return (self.gamma * weight_value).contiguous()


class AttentionAggregationModule(nn.Module):
    def __init__(self, in_chan, out_chan):
        super(AttentionAggregationModule, self).__init__()
        self.convblk = ConvBnRelu(in_chan, out_chan, ksize=1, stride=1, pad=0)
        self.conv_atten = Attention(out_chan)

    def forward(self, x):
        fcat = torch.cat(x, dim=1)
        feat = self.convblk(fcat)
        atten = self.conv_atten(feat)
        feat_out = atten + feat
        return feat_out
"""
    From RSHazeNet
"""
class RSHazeNet(nn.Module):
    def __init__(self, in_chans=3, out_chans=3, dim=32, depths=(2, 3, 4), dwt_filter = 4, class_num=6):
        super(RSHazeNet, self).__init__()
        self.net_depths=sum(depths)
        self.patch_embed_level_1 = OverlapPatchEmbed(
            in_c=in_chans, 
            embed_dim=dim, 
            bias=False
        )
        self.patch_unembed_level_1 = nn.Sequential(
            *[ConvBlock(self.net_depths, dim) for i in range(depths[0])]
        )
        
        self.skip_connection_level_1_pre = nn.Sequential(
            *[ConvBlock(self.net_depths, dim) for i in range(depths[0])]
        )
        self.skip_connection_level_1_post = nn.Sequential(
            *[ConvBlock(self.net_depths, dim) for i in range(depths[0])]
        )

        self.down_level_2 = Downsample(dim)
        self.skip_connection_level_2_pre = nn.Sequential(
            *[ConvBlock(self.net_depths, int(dim * 2 ** 1)) for i in range(depths[1])]
        )
        self.skip_connection_level_2_mid = nn.Sequential(
            *[ConvBlock(self.net_depths, int(dim * 2 ** 1)) for i in range(depths[1])]
        )
        self.skip_connection_level_2_post = nn.Sequential(
            *[ConvBlock(self.net_depths, int(dim * 2 ** 1)) for i in range(depths[1])]
        )
        
        self.down_level_3 = Downsample(int(dim * 2 ** 1))
        self.skip_connection_level_3_pre = nn.Sequential(
            *[ConvBlock(self.net_depths, int(dim * 2 ** 2)) for i in range(depths[2])]
        )
        self.skip_connection_level_3_post = nn.Sequential(
            *[ConvBlock(self.net_depths, int(dim * 2 ** 2)) for i in range(depths[2])]
        )
        
        self.cmfi_level_1_2 = CMFI(dim)
        self.cmfi_level_2_3 = CMFI(int(dim * 2 ** 1))
        
        self.decode3 = DecoderBlock(
            net_depths = self.net_depths, 
            depth = depths[1], 
            dim = int(dim * 2 ** 2),
            upsample_iterate = 2
        )

        self.decode2 = DecoderBlock(
            net_depths = self.net_depths, 
            depth = depths[1], 
            dim = int(dim * 2 ** 1),
            upsample_iterate = 1
        )
        
        self.decode1 = DecoderBlock(
            net_depths = self.net_depths, 
            depth = depths[0], 
            dim = int(dim * 2 ** 0),
            upsample_iterate = 0
        )
        self.fusion = SKFusion(
            dim=32, 
            height=3, 
            reduction=8
        )
        self.output_level_1 = OverlapPatchUnembed(
            dim, 
            out_chans, 
            kernel_size=3, 
            bias=False
        )

        
        # Segmentation
        self.decode3S = DecoderBlock(
            net_depths = self.net_depths, 
            depth = depths[1], 
            dim = int(dim * 2 ** 2),
            upsample_iterate = 2
        )
        self.decode2S = DecoderBlock(
            net_depths = self.net_depths, 
            depth = depths[1], 
            dim = int(dim * 2 ** 1),
            upsample_iterate = 1
        )
        self.decode1S = DecoderBlock(
            net_depths = self.net_depths, 
            depth = depths[0], 
            dim = int(dim * 2 ** 0),
            upsample_iterate = 0
        )
        self.fusionS = AttentionAggregationModule(
            dim * 3, 
            dim * 3
        )
        self.dropout = nn.Dropout2d(p=0.2, inplace=True)
        self.output_level_1S = OverlapPatchUnembed(
            dim * 3, 
            class_num, 
            kernel_size=1, 
            bias=False
        )
        
        self.sigma = nn.Parameter(torch.zeros(2, dtype=torch.float32), requires_grad=True)

    def forward_features(self, x):
        # Encoder
        x = self.patch_embed_level_1(x)
        skip_level_1_pre = self.skip_connection_level_1_pre(x)
        
        x = self.down_level_2(skip_level_1_pre)
        skip_level_2_pre = self.skip_connection_level_2_pre(x)

        x = self.down_level_3(skip_level_2_pre)
        latent_pre = self.skip_connection_level_3_pre(x)

        skip_level_2_mid, latent_post = self.cmfi_level_2_3(skip_level_2_pre, latent_pre)
        skip_level_2_mid = self.skip_connection_level_2_mid(skip_level_2_mid)
        skip_level_1_mid, skip_level_2_mid = self.cmfi_level_1_2(skip_level_1_pre, skip_level_2_mid)
        skip_level_2_mid = self.skip_connection_level_2_post(skip_level_2_mid)
        skip_level_1_mid = self.skip_connection_level_1_post(skip_level_1_mid)

        latent_post = self.skip_connection_level_3_post(latent_post)
        
        x_3 = self.decode3(x=latent_post)
        x_2 = self.decode2(x=skip_level_2_mid)
        x_1 = self.decode1(x=skip_level_1_mid)
        
        x = self.fusion([x_1, x_2, x_3])
        x = self.patch_unembed_level_1(x)
        x = self.output_level_1(x)
        
        
        x_3S = self.decode3S(x=latent_post)
        x_2S = self.decode2S(x=skip_level_2_mid)
        x_1S = self.decode1S(x=skip_level_1_mid)
        
        x_S = self.fusionS([x_3S, x_2S, x_1S])
        x_S = self.dropout(x_S)
        x_S = self.output_level_1S(x_S)
        
        return x, x_S

    def forward(self, x, loss_params=False):
        input_ = x
        _, _, h, w = input_.shape

        x, x_S = self.forward_features(x)
        K, B = torch.split(x, [1, 3], dim=1)
        x = K * input_ - B + input_
        x = x[:, :, :h, :w]
        
        if loss_params:
            return x, x_S, self.sigma
        
        return x, x_S