import torch
import torch.nn as nn
from net.common import OverlapPatchEmbed, Downsample, CMFI, ConvBlock, DecoderBlock

class RSHazeNet(nn.Module):
    def __init__(self, in_chans=3, out_chans=3, dim=32, depths=(2, 3, 4)):
        super(RSHazeNet, self).__init__()
        
        self.net_depths=sum(depths)
        self.patch_embed_level_1 = OverlapPatchEmbed(
            in_c=in_chans, 
            embed_dim=dim, 
            bias=False
        )
        self.patch_unembed_level_1 = OverlapPatchEmbed(
            in_c=dim*3, 
            embed_dim=4, 
            bias=False
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

    def forward_features(self, x):
        # high_structures = []
        input_ = x
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
        
        x = torch.cat([x_1, x_2, x_3], dim=1)
        x = self.patch_unembed_level_1(x)
        return x

    def forward(self, x):
        input_ = x
        _, _, h, w = input_.shape

        x = self.forward_features(x)
        K, B = torch.split(x, [1, 3], dim=1)
        x = K * input_ - B + input_
        x = x[:, :, :h, :w]
        return x