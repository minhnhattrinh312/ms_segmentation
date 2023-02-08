import torch
import torch.nn as nn
from timm.models.layers import DropPath
import timm.optim
import torch.optim as optim
import pytorch_lightning as pl
from segment3d.losses import ActiveFocalLoss
from segment3d.metrics import dice_MS

class LayerNorm(nn.Sequential):
    def __init__(self, normalized_shape):
        super().__init__()
        # self.add_module("layer_norm", nn.BatchNorm3d(normalized_shape))
        self.add_module("layer_norm", nn.InstanceNorm3d(normalized_shape, affine=True))

class Block(nn.Module):
    def __init__(self, dim, drop_path_rate=0., layer_scale_init_value=1):
        super().__init__()
        self.dwconv = nn.Conv3d(dim, dim, kernel_size=7, padding=3, groups=dim) # depthwise conv
        self.norm = LayerNorm(dim, data_format="channels_last")
        self.pwconv1 = nn.Linear(dim, 4 * dim) # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELUà()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim)), 
                                    requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0. else nn.Identity()

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 4, 1) # (N, C, H, W, D) -> (N, H, W, D, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, -1, 1, 2, 3) # (N, H, W, D, C) -> (N, C, H, W, D)

        x = input + self.drop_path(x)
        return x

class ConvBlock(nn.Module):
    def __init__(self, in_dim, drop_path_rate=0, layer_scale_init_value=1):
        super().__init__()
        self.conv_layer = nn.Sequential(
            nn.Conv3d(in_dim, in_dim, kernel_size=7, padding=3, groups=in_dim),
            LayerNorm(in_dim),
            # nn.GELU(),
            nn.Conv3d(in_dim, 4 * in_dim, kernel_size=1, padding=0),
            # LayerNorm(4 * in_dim),
            nn.GELU(),
            nn.Conv3d(4 * in_dim, in_dim, kernel_size=1, padding=0),
            # LayerNorm(in_dim),
            # CBAM(in_dim)
            )
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((in_dim, 1, 1, 1)), 
                                  requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0 else nn.Identity()
    def forward(self, x):
        res = x
        x = self.conv_layer(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = res + self.drop_path(x)
        return x

class ConvBlock_vs2(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.conv_layer = nn.Sequential(
            nn.Conv3d(in_dim, in_dim, kernel_size=7, padding=3, groups=in_dim),
            LayerNorm(in_dim),
            nn.GELU(),
            nn.Conv3d(in_dim, 4 * in_dim, kernel_size=1, padding=0),
            LayerNorm(4 * in_dim),
            nn.GELU(),
            nn.Conv3d(4 * in_dim, out_dim, kernel_size=1, padding=0),
            LayerNorm(out_dim),
            nn.GELU(),
            )
    def forward(self, x):
        x = self.conv_layer(x)
        return x
    
class DecoderBlock(nn.Module):
    def __init__(self, in_dim, skip_dim, out_dim):
        super().__init__()
        self.upsample = nn.ConvTranspose3d(in_dim, out_dim, kernel_size=2, stride=2)
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
    def __init__(self, in_dim, num_class, depths=[3, 3, 9, 3], dim_featrue=48, drop_path_rate=0.5):
        super().__init__()
        dims_encoder = list(map(lambda x: dim_featrue * x, [1, 2, 4, 8]))
        ######################## encoder ##################################################
        self.downsample_layers = nn.ModuleList() # stem and 3 intermediate downsampling conv layers
        stem = nn.Sequential(
            nn.Conv3d(in_dim, dims_encoder[0], kernel_size=4, stride=4),
            LayerNorm(dims_encoder[0])
        )
        self.downsample_layers.append(stem)
        for i in range(3):
            downsample_layer = nn.Sequential(
                    LayerNorm(dims_encoder[i]),
                    nn.Conv3d(dims_encoder[i], dims_encoder[i+1], kernel_size=2, stride=2),
            )
            self.downsample_layers.append(downsample_layer)

        self.stages = nn.ModuleList() # 4 feature resolution stages, each consisting of multiple residual blocks
        dp_rates=[x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))] 
        cur = 0
        for i in range(4):
            stage = nn.Sequential(
                *[ConvBlock(dims_encoder[i], dp_rates[cur + j]) for j in range(depths[i])]
            )
            self.stages.append(stage)
            cur += depths[i]

        ########### decoder #########################################
        dim_featrue //= 2
        dims_decoder = list(map(lambda x: dim_featrue * x, [8, 4, 2, 1]))
        self.upsample_layers = nn.ModuleList()
        for i in range(3):
            stage = DecoderBlock(dims_encoder[-i-1], dims_encoder[-i-2], dims_decoder[i])
            self.upsample_layers.append(stage)
        i += 1
        self.final_conv = nn.Sequential(
            nn.ConvTranspose3d(dims_decoder[i-1], dims_decoder[i], kernel_size=4, stride=4),
            ConvBlock_vs2(dims_decoder[i], dims_decoder[i]),
            LayerNorm(dims_decoder[i]),
            nn.GELU(),
            nn.Conv3d(dims_decoder[i], num_class, kernel_size=1, padding=0),
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


class Segmentor(pl.LightningModule):
    def __init__(self, model, class_weight, num_classes, 
                 learning_rate, factor_lr, patience_lr):
        super().__init__()
        self.model=model
        self.class_weight=class_weight
        self.num_classes=num_classes
        self.learning_rate=learning_rate
        self.factor_lr = factor_lr
        self.patience_lr= patience_lr

    def forward(self, x):
        return self.model(x)

    def _step(self, batch):
        image, y_true = batch
        y_pred = self.model(image)
        loss = ActiveFocalLoss(self.device, self.class_weight, self.num_classes)(y_true, y_pred)
        dice_ms = dice_MS(y_true, y_pred)
        return loss, dice_ms

    def training_step(self, batch, batch_idx):
        loss, dice_ms = self._step(batch)
        metrics = {"losses": loss, "diceTrain": dice_ms}
        self.log_dict(metrics, on_step=True, on_epoch=True, prog_bar = True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, dice_ms = self._step(batch)
        metrics = {"losses": loss, "diceVal": dice_ms}
        self.log_dict(metrics, prog_bar = True)
        return metrics

    def test_step(self, batch, batch_idx):
        loss, dice_ms = self._step(batch)
        metrics = {"losses": loss, "diceTest": dice_ms}
        self.log_dict(metrics, prog_bar = True)
        return metrics

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        x, y = batch
        y_hat = self.model(x)
        return y_hat.cpu().numpy()

    def configure_optimizers(self):
        optimizer = timm.optim.Nadam(self.parameters(), lr=self.learning_rate)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 
                                                         mode="max", factor=self.factor_lr, 
                                                         patience=self.patience_lr, verbose =True)
        lr_schedulers = {"scheduler": scheduler, "monitor": "diceVal"}
        # return [optimizer]
        return [optimizer], lr_schedulers
