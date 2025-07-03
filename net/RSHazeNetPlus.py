import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
cudnn.benchmark = True

from torchsummary import summary
from timm.models.layers import to_2tuple
from thop import profile, clever_format

from net.common import IFTE, OverlapPatchEmbed, Downsample, Upsample, CMFI, ConvBlock

class RSHazeNet(nn.Module):
    def __init__(self, in_chans=3, out_chans=3, dim=32, depths=(2, 3, 4)):
        super(RSHazeNet, self).__init__()
        
        self.net_depths=sum(depths)
        self.patch_embed_level_1 = OverlapPatchEmbed(
            in_c=in_chans, 
            embed_dim=dim, 
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
        
        self.up_level_3 = Upsample(int(dim * 2 ** 2))
        self.up_level_2 = Upsample(int(dim * 2 ** 1))

        self.ifte_level_2 = IFTE(int(dim * 2 ** 1))
        self.ifte_level_1 = IFTE(dim)

        self.output_level_1 = nn.Conv2d(
            dim, 
            out_chans, 
            kernel_size=to_2tuple(3), 
            padding=1, 
            padding_mode='reflect', 
            bias=False
        )

    def forward_features(self, x):
        # Encoder
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
    
if __name__=="__main__":
    image_size = (256, 256, 3)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    model = RSHazeNet(
        in_chans=3, 
        out_chans=4,
        dim=32,
        depths=(2, 2, 2)
    ).to(device)
    macs, params = profile(model, inputs=(torch.rand(image_size[::-1])[None, ...].to(device), ))
    macs, params = clever_format([macs, params], "%.3f")
    print('macs', macs)
    print('params', params)
    summary(model, image_size[::-1], device=device)
