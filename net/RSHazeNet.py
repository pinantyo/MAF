import torch
import torch.nn as nn
import torch.nn.functional as F

from timm.models.layers import to_2tuple
from net.common import IFTE, OverlapPatchEmbed, Downsample, Upsample, CMFI


class BasicBlock(nn.Module):
    def __init__(self, dim, division_ratio=4):
        super(BasicBlock, self).__init__()
        self.dim = dim
        self.dim_partial = int(dim // division_ratio)
        hidden_features = int(dim * 4)

        self.conv_1 = nn.Conv2d(self.dim_partial, self.dim_partial, kernel_size=to_2tuple(1))
        self.conv_3 = nn.Conv2d(self.dim_partial, self.dim_partial, kernel_size=to_2tuple(3), padding=3, dilation=3, groups=self.dim_partial)
        self.conv_5 = nn.Conv2d(self.dim_partial, self.dim_partial, kernel_size=to_2tuple(5), padding=6, dilation=3, groups=self.dim_partial)
        self.conv_7 = nn.Conv2d(self.dim_partial, self.dim_partial, kernel_size=to_2tuple(7), padding=9, dilation=3, groups=self.dim_partial)

        self.mlp = self.mlp = nn.Sequential(
            nn.Conv2d(dim, hidden_features, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(hidden_features, dim, 1, bias=False)
        )

        layer_scale_init_value = 0.
        self.layer_scale = nn.Parameter(layer_scale_init_value * torch.ones((dim)), requires_grad=True)

    def forward(self, x):
        input_ = x
        x_1, x_2, x_3, x_4 = torch.split(x, [self.dim_partial, self.dim_partial, self.dim_partial, self.dim_partial], dim=1)
        x_1 = self.conv_1(x_1)
        x_2 = self.conv_3(x_2)
        x_3 = self.conv_5(x_3)
        x_4 = self.conv_7(x_4)

        x = torch.cat((x_1, x_2, x_3, x_4), 1)
        x = self.layer_scale.unsqueeze(-1).unsqueeze(-1) * self.mlp(x) + input_

        return x


class RSHazeNet(nn.Module):
    def __init__(self, in_chans=3, out_chans=4, dim=32, depths=(2, 3, 4)):
        super(RSHazeNet, self).__init__()

        self.patch_embed_level_1 = OverlapPatchEmbed(in_c=in_chans, embed_dim=dim, bias=False)
        self.skip_connection_level_1_pre = nn.Sequential(*[BasicBlock(dim) for _ in range(depths[0] // 2)])

        self.skip_connection_level_1_post = nn.Sequential(*[BasicBlock(dim) for _ in range(depths[0] // 2)])

        self.down_level_2 = Downsample(dim)
        self.skip_connection_level_2_pre = nn.Sequential(*[BasicBlock(int(dim * 2 ** 1)) for _ in range(depths[1] // 3)])

        self.skip_connection_level_2_mid = nn.Sequential(*[BasicBlock(int(dim * 2 ** 1)) for _ in range(depths[1] // 3)])

        self.skip_connection_level_2_post = nn.Sequential(*[BasicBlock(int(dim * 2 ** 1)) for _ in range(depths[1] // 3)])

        self.down_level_3 = Downsample(int(dim * 2 ** 1))
        self.skip_connection_level_3_pre = nn.Sequential(*[BasicBlock(int(dim * 2 ** 2)) for _ in range(depths[2] // 2)])

        self.skip_connection_level_3_post = nn.Sequential(*[BasicBlock(int(dim * 2 ** 2)) for _ in range(depths[2] // 2)])

        self.up_level_3 = Upsample(int(dim * 2 ** 2))
        self.up_level_2 = Upsample(int(dim * 2 ** 1))

        self.cmfi_level_1_2 = CMFI(dim)
        self.cmfi_level_2_3 = CMFI(int(dim * 2 ** 1))
        self.ifte_level_2 = IFTE(int(dim * 2 ** 1))
        self.ifte_level_1 = IFTE(dim)

        self.output_level_1 = nn.Conv2d(dim, out_chans, kernel_size=to_2tuple(3), padding=1, padding_mode='reflect', bias=False)

    def forward_features(self, x):
        x = self.patch_embed_level_1(x)
        skip_level_1_pre = self.skip_connection_level_1_pre(x)

        x = self.down_level_2(x)
        skip_level_2_pre = self.skip_connection_level_2_pre(x)

        x = self.down_level_3(x)
        latent_pre = self.skip_connection_level_3_pre(x)

        skip_level_2_pre, latent_pre = self.cmfi_level_2_3(skip_level_2_pre, latent_pre)

        skip_level_2_mid = self.skip_connection_level_2_mid(skip_level_2_pre)

        skip_level_1_pre, skip_level_2_mid = self.cmfi_level_1_2(skip_level_1_pre, skip_level_2_mid)

        skip_level_2 = self.skip_connection_level_2_post(skip_level_2_mid)
        skip_level_1 = self.skip_connection_level_1_post(skip_level_1_pre)

        latent_post = self.skip_connection_level_3_post(latent_pre)
        x = self.up_level_3(latent_post)

        x = self.ifte_level_2([x, skip_level_2]) + x
        x = self.up_level_2(x)

        x = self.ifte_level_1([x, skip_level_1]) + x
        x = self.output_level_1(x)
        return x

    def forward(self, x):
        input_ = x
        _, _, h, w = input_.shape

        x = self.forward_features(x)
        K, B = torch.split(x, [1, 3], dim=1)

        x = K * input_ - B + input_
        x = x[:, :, :h, :w]

        return x