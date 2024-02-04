import torch
import torch.nn as nn
from timm.models.layers import DropPath
convnext_urls = {
    "convnext_tiny_1k": "https://dl.fbaipublicfiles.com/convnext/convnext_tiny_1k_224_ema.pth",
}
class LayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_first"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError 
        self.normalized_shape = (normalized_shape, )
    
    def forward(self, x):
        if self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x
        elif self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)

class Block(nn.Module):
    def __init__(self, dim, drop_path_rate=0., layer_scale_init_value=1):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim) # depthwise conv
        self.norm = LayerNorm(dim, data_format="channels_last")
        self.pwconv1 = nn.Linear(dim, 4 * dim) # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim)), 
                                    requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0. else nn.Identity()

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1) # (N, C, H, W) -> (N, H, W, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2) # (N, H, W, C) -> (N, C, H, W)

        x = input + self.drop_path(x)
        return x

class ConvBlock_vs2(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.conv_layer = nn.Sequential(
            nn.Conv2d(in_dim, in_dim, kernel_size=7, padding=3, groups=in_dim),
            LayerNorm(in_dim),
            nn.GELU(),
            nn.Conv2d(in_dim, 4 * in_dim, kernel_size=1, padding=0),
            LayerNorm(4 * in_dim),
            nn.GELU(),
            nn.Conv2d(4 * in_dim, out_dim, kernel_size=1, padding=0),
            LayerNorm(out_dim),
            nn.GELU(),
            )
    def forward(self, x):
        x = self.conv_layer(x)
        return x
class DecoderBlock(nn.Module):
    def __init__(self, in_dim, skip_dim, out_dim):
        super().__init__()
        self.upsample = nn.ConvTranspose2d(in_dim, out_dim, kernel_size=2, stride=2)
        self.attention = ConvBlock_vs2(skip_dim, skip_dim // 2)
        indim = out_dim + skip_dim // 2
        self.conv_layer = ConvBlock_vs2(indim, out_dim)
    def forward(self, x, skip):
        output = self.upsample(x)
        attention = self.attention(skip)
        output = torch.cat([output, attention], dim=1)
        output = self.conv_layer(output)
        return output

class SkipNet(nn.Module):
    def __init__(self, in_dim=3, num_class=2, depths=[3, 3, 9, 3], dims_encoder=[96, 192, 384, 768], drop_path_rate=0.5):
        super().__init__()
        ######################## encoder ##################################################
        self.downsample_layers = nn.ModuleList() # stem and 3 intermediate downsampling conv layers
        stem = nn.Sequential(
            nn.Conv2d(in_dim, dims_encoder[0], kernel_size=4, stride=4),
            LayerNorm(dims_encoder[0], eps=1e-6, data_format="channels_first")
        )
        self.downsample_layers.append(stem)
        for i in range(3):
            downsample_layer = nn.Sequential(
                    LayerNorm(dims_encoder[i], eps=1e-6, data_format="channels_first"),
                    nn.Conv2d(dims_encoder[i], dims_encoder[i+1], kernel_size=2, stride=2),
            )
            self.downsample_layers.append(downsample_layer)

        self.stages = nn.ModuleList() # 4 feature resolution stages, each consisting of multiple residual blocks
        dp_rates=[x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))] 
        cur = 0
        for i in range(4):
            stage = nn.Sequential(
                *[Block(dims_encoder[i], dp_rates[cur + j]) for j in range(depths[i])]
            )
            self.stages.append(stage)
            cur += depths[i]

        ########### decoder #########################################
        dims_decoder = list(map(lambda x: x//2, dims_encoder[::-1]))
        self.upsample_layers = nn.ModuleList()
        for i in range(3):
            stage = DecoderBlock(dims_encoder[-i-1], dims_encoder[-i-2], dims_decoder[i])
            self.upsample_layers.append(stage)
        i += 1
        self.final_conv = nn.Sequential(
            nn.ConvTranspose2d(dims_decoder[i-1], dims_decoder[i], kernel_size=4, stride=4),
            ConvBlock_vs2(dims_decoder[i], dims_decoder[i]),
            LayerNorm(dims_decoder[i]),
            nn.GELU(),
            nn.Conv2d(dims_decoder[i], num_class, kernel_size=1, padding=0),
            nn.Softmax(dim=1)
            )
        
    
    def forward(self, x):
        encoder_for_cat = []
        output_cat = []
        for i in range(3):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
            encoder_for_cat.append(x)
        i += 1
        x = self.downsample_layers[i](x)
        x = self.stages[i](x)
        for i in range(3):
            skip_connection = encoder_for_cat.pop()
            x = self.upsample_layers[i](x, skip_connection)

        output = self.final_conv(x)
        
        return output