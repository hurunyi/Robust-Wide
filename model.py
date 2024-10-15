import torch
from torch import nn
import torch.nn.functional as F
import numpy as np


def GroupNorm32(channels):
    return nn.GroupNorm(32, channels)


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None, norm_type="batch"):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels

        if norm_type == "batch":
            normalization = nn.BatchNorm2d
        elif norm_type == "instance":
            normalization = nn.InstanceNorm2d
        elif norm_type == "group":
            normalization = GroupNorm32
        else:
            raise NotImplementedError

        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            normalization(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            normalization(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    def __init__(self, in_channels, out_channels, norm_type="batch"):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels, norm_type=norm_type)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=True, norm_type="batch"):
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2, norm_type=norm_type)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels, norm_type=norm_type)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):
    def __init__(self, n_channels=4, n_classes=3, bilinear=False, norm_type="batch"):
        super().__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = (DoubleConv(n_channels, 64, norm_type=norm_type))
        self.down1 = (Down(64, 128, norm_type=norm_type))
        self.down2 = (Down(128, 256, norm_type=norm_type))
        self.down3 = (Down(256, 512, norm_type=norm_type))
        factor = 2 if bilinear else 1
        self.down4 = (Down(512, 1024 // factor, norm_type=norm_type))
        self.up1 = (Up(1024, 512 // factor, bilinear, norm_type=norm_type))
        self.up2 = (Up(512, 256 // factor, bilinear, norm_type=norm_type))
        self.up3 = (Up(256, 128 // factor, bilinear, norm_type=norm_type))
        self.up4 = (Up(128, 64, bilinear, norm_type=norm_type))
        self.outc = (OutConv(64, n_classes))

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.outc(x)
        return x


class Interpolate(nn.Module):
    def __init__(self, size=None, scale_factor=None):
        super(Interpolate, self).__init__()
        self.function = F.interpolate
        self.size = size
        self.scale_factor = scale_factor

    def forward(self, x):
        return self.function(x, size=self.size, scale_factor=self.scale_factor)


class ConvTBNRelu(nn.Module):
    def __init__(self, channels_in, channels_out, stride=2):
        super(ConvTBNRelu, self).__init__()

        self.layers = nn.Sequential(
            nn.ConvTranspose2d(channels_in, channels_out, kernel_size=2, stride=stride, padding=0),
            nn.BatchNorm2d(channels_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.layers(x)


class ExpandNet(nn.Module):
    def __init__(self, in_channels, out_channels, blocks):
        super(ExpandNet, self).__init__()

        layers = [ConvTBNRelu(in_channels, out_channels)] if blocks != 0 else []
        for _ in range(blocks - 1):
            layer = ConvTBNRelu(out_channels, out_channels)
            layers.append(layer)

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class EncoderUnet(nn.Module):
    def __init__(self, image_size=64, message_length=16, in_channels=4, channels=64, norm_type="batch", final_skip=False):
        super(EncoderUnet, self).__init__()

        message_convT_blocks = int(np.log2(image_size // int(np.sqrt(message_length))))
        interpolate_scale = 1
        max_blocks_num = 6
        if message_convT_blocks > max_blocks_num:
            interpolate_scale = 2 ** (message_convT_blocks - max_blocks_num)
            message_convT_blocks = max_blocks_num

        if norm_type == "batch":
            normalization = nn.BatchNorm2d
        elif norm_type == "instance":
            normalization = nn.InstanceNorm2d
        elif norm_type == "group":
            normalization = GroupNorm32
        else:
            raise NotImplementedError

        self.message_pre_layer = nn.Sequential(
            nn.Conv2d(1, channels, 3, 1, padding=1),
            normalization(channels),
            nn.ReLU(inplace=True),
            ExpandNet(channels, channels, blocks=message_convT_blocks),
        )
        if interpolate_scale != 1:
            self.message_pre_layer.add_module(
                name="interpolate_layer",
                module=Interpolate(scale_factor=interpolate_scale)
            )

        self.final_skip = final_skip

        if final_skip:
            self.unet = UNet(n_channels=in_channels + channels, n_classes=channels, norm_type=norm_type)
            self.final_layer = nn.Conv2d(channels + in_channels, in_channels, kernel_size=1)
        else:
            self.unet = UNet(n_channels=in_channels + channels, n_classes=in_channels, norm_type=norm_type)

    def forward(self, image, message):
        # Message Processor
        size = int(np.sqrt(message.shape[1]))
        message_image = message.view(-1, 1, size, size)
        message_pre = self.message_pre_layer(message_image)
        # concatenate
        concat = torch.cat([image, message_pre], dim=1)
        output = self.unet(concat)

        if self.final_skip:
            # final skip connection
            concat_final = torch.cat([output, image], dim=1)
            # last Conv part of Encoder
            output = self.final_layer(concat_final)

        return output


class ResidualBlock(nn.Module):
    def __init__(self, in_channel, channel_num, dilation=1, group=1, norm_type="batch"):
        super(ResidualBlock, self).__init__()

        if norm_type == "batch":
            normalization = nn.BatchNorm2d
        elif norm_type == "instance":
            normalization = nn.InstanceNorm2d
        elif norm_type == "group":
            normalization = GroupNorm32
        else:
            raise NotImplementedError

        self.conv1 = nn.Conv2d(in_channel, channel_num, 3, 1, padding=dilation, dilation=dilation, groups=group,
                               bias=False)
        self.norm1 = normalization(channel_num)
        self.conv2 = nn.Conv2d(channel_num, channel_num, 3, 1, padding=dilation, dilation=dilation, groups=group,
                               bias=False)
        self.norm2 = normalization(channel_num)

    def forward(self, x):
        y = F.relu(self.norm1(self.conv1(x)))
        y = self.norm2(self.conv2(y))
        return F.relu(x + y)


class DecoderResnet(nn.Module):
    def __init__(self, image_size=64, message_length=16, in_channels=3, norm_type="batch"):
        super(DecoderResnet, self).__init__()
        stride_blocks = int(np.log2(image_size // int(np.sqrt(message_length))))
        interpolate_scale = 1
        max_blocks_num = 6
        if stride_blocks > max_blocks_num:
            interpolate_scale = 2 ** (max_blocks_num - stride_blocks)
            stride_blocks = max_blocks_num

        if norm_type == "batch":
            normalization = nn.BatchNorm2d
        elif norm_type == "instance":
            normalization = nn.InstanceNorm2d
        elif norm_type == "group":
            normalization = GroupNorm32
        else:
            raise NotImplementedError

        self.first_layers = nn.Sequential(
            nn.Conv2d(in_channels, 128, 3, 1, 1, bias=False),
            normalization(128),
            nn.ReLU(inplace=True),
        )

        self.down_layers = nn.ModuleList([nn.Sequential(
            nn.Conv2d(128 * (2 ** i), 128 * (2 ** (i+1)), 3, 2, 1, bias=False),
            normalization(128 * (2 ** (i+1))),
            nn.ReLU(inplace=True),
        ) for i in range(stride_blocks)])

        self.down_layers.append(nn.Sequential(
            nn.Conv2d(128 * (2 ** stride_blocks), 128, 3, 1, 1, bias=False),
            normalization(128),
            nn.ReLU(inplace=True)
        ))
        if interpolate_scale != 1:
            self.down_layers.append(Interpolate(scale_factor=interpolate_scale))

        self.keep_layers = nn.Sequential(
            ResidualBlock(128, 128, dilation=2, norm_type=norm_type),
            ResidualBlock(128, 128, dilation=2, norm_type=norm_type),
            ResidualBlock(128, 128, dilation=2, norm_type=norm_type),
            ResidualBlock(128, 128, dilation=2, norm_type=norm_type),
            ResidualBlock(128, 128, dilation=4, norm_type=norm_type),
            ResidualBlock(128, 128, dilation=4, norm_type=norm_type),
            ResidualBlock(128, 128, dilation=4, norm_type=norm_type),
            ResidualBlock(128, 128, dilation=4, norm_type=norm_type),
            ResidualBlock(128, 128, dilation=1, norm_type=norm_type),
        )

        self.final_layers = nn.Sequential(
            nn.Conv2d(128, 1, 3, 1, 1, bias=False),
            normalization(1) if norm_type != "group" else nn.GroupNorm(1, 1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        y = self.first_layers(x)
        for down_layer in self.down_layers:
            y = down_layer(y)
        y = self.keep_layers(y)
        y = self.final_layers(y)
        y = y.view(y.shape[0], -1)
        return y


class WatermarkModel(nn.Module):
    def __init__(
        self,
        wm_enc_config,
        wm_dec_config,
        device="cuda",
        weight_dtype=torch.float32
    ):
        super(WatermarkModel, self).__init__()

        self.encoder = EncoderUnet(**wm_enc_config)
        self.decoder = DecoderResnet(**wm_dec_config)
        self.wm_enc_config = wm_enc_config
        self.wm_dec_config = wm_dec_config
        self.weight_dtype = weight_dtype
        self.device = device
