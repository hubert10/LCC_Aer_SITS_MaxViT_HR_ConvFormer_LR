import numpy as np
import torch
import torch.nn as nn
from torch.nn.functional import interpolate


class FM_cropped(nn.Module):
    """
    Conv. + interpolation layers to resize featmaps
    """

    def __init__(self, in_channels, out_channels_list):
        super(FM_cropped, self).__init__()
        self.conv_list = nn.ModuleList()

        for out_ch_size in out_channels_list:
            self.conv_list.append(
                nn.Conv2d(
                    in_channels=in_channels, out_channels=out_ch_size, kernel_size=1
                )
            )

    def forward(self, mask, height_list):
        # print()
        # print('FM_cropped:::::::::::::::::::::::::', mask.shape)
        # print('height_list:::::::::::::::::::::::::', height_list)

        out_mask = []
        for nb, size in enumerate(height_list):
            out_mask.append(
                interpolate(
                    self.conv_list[nb](mask), size=height_list[nb], mode="bilinear"
                )
            )
        return out_mask


class FM_collapsed(nn.Module):
    """
    MLPs to summarize spatial dim of featmaps
    """

    def __init__(self, in_channels, out_channels_list):
        super(FM_collapsed, self).__init__()

        self.mlps = nn.ModuleList()
        for out_ch_size in out_channels_list:
            layers = []
            m_value = int(np.abs(in_channels - out_ch_size) / 3)  ## 3 layers
            if in_channels < out_ch_size:
                layers.extend(
                    [
                        nn.Linear(in_channels, in_channels + m_value * 1),
                        nn.Dropout(0.3),
                        nn.ReLU(inplace=True),
                        nn.Linear(in_channels + m_value * 1, in_channels + m_value * 2),
                        nn.Dropout(0.3),
                        nn.ReLU(inplace=True),
                        nn.Linear(in_channels + m_value * 2, out_ch_size),
                    ]
                )
                self.mlps.append(nn.Sequential(*layers))
            elif in_channels > out_ch_size:
                layers.extend(
                    [
                        nn.Linear(in_channels, in_channels - m_value * 1),
                        nn.Dropout(0.3),
                        nn.ReLU(inplace=True),
                        nn.Linear(in_channels - m_value * 1, in_channels - m_value * 2),
                        nn.Dropout(0.3),
                        nn.ReLU(inplace=True),
                        nn.Linear(in_channels - m_value * 2, out_ch_size),
                    ]
                )
                self.mlps.append(nn.Sequential(*layers))
            else:
                layers.extend(
                    [
                        nn.Linear(in_channels, in_channels * 2),
                        nn.Dropout(0.3),
                        nn.ReLU(inplace=True),
                        nn.Linear(in_channels * 2, in_channels * 2),
                        nn.Dropout(0.3),
                        nn.ReLU(inplace=True),
                        nn.Linear(in_channels * 2, out_ch_size),
                    ]
                )
                self.mlps.append(nn.Sequential(*layers))

    def forward(self, fmap, height_list):
        out = []
        for mlp, h in zip(self.mlps, height_list):
            channels_ = mlp(fmap)
            out.append(
                torch.unsqueeze(
                    channels_.view(channels_.size()[0], 1)
                    .view(channels_.size()[0], 1, 1)
                    .repeat(1, h, h),
                    0,
                )
            )

        return out


#### IF USING AERIAL METADATA
class mtd_encoding_mlp(nn.Module):
    """
    Light MLP to format aertial metadata
    """

    def __init__(self, in_, out_):
        super(mtd_encoding_mlp, self).__init__()

        self.enc_mlp = nn.Sequential(
            nn.Linear(in_, 64),
            nn.Dropout(0.5),
            nn.ReLU(inplace=True),
            nn.Linear(64, 32),
            nn.Dropout(0.5),
            nn.ReLU(inplace=True),
            nn.Linear(32, out_),
        )

    def forward(self, x):
        x = self.enc_mlp(x)
        return x


class UpSamplingSITS(nn.Module):
    """
    IN : torch.Size([4, 64, 10, 10])
    layer: ConvTranspose2d(64, 512, kernel_size=(7, 7), stride=(1, 1), bias=False)
    x OUT: torch.Size([4, 512, 16, 16])

    x IN : torch.Size([4, 512, 16, 16])
    layer: ConvTranspose2d(512, 256, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False)
    x OUT: torch.Size([4, 256, 32, 32])

    x IN : torch.Size([4, 256, 32, 32])
    layer: ConvTranspose2d(256, 128, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False)
    x OUT: torch.Size([4, 128, 64, 64])

    x IN : torch.Size([4, 128, 64, 64])
    layer: ConvTranspose2d(128, 64, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False)
    x OUT: torch.Size([4, 64, 128, 128])

    x IN : torch.Size([4, 64, 128, 128])
    layer: ConvTranspose2d(64, 64, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False)
    x OUT: torch.Size([4, 64, 256, 256])

    x IN : torch.Size([4, 64, 256, 256])
    layer: ConvTranspose2d(64, 5, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False)
    x OUT: torch.Size([4, 5, 512, 512])
    """

    def __init__(self, in_channels, out_channels_list):
        super(UpSamplingSITS, self).__init__()

        # self.channels = channels
        # self.image_size = image_size

        self.main = nn.Sequential(
            # Layer 1
            nn.ConvTranspose2d(in_channels, out_channels_list[5], 7, 1, 0, bias=False),
            # nn.BatchNorm2d(512),
            # nn.ReLU(True),
            # Layer 2
            nn.ConvTranspose2d(
                out_channels_list[5], out_channels_list[4], 4, 2, 1, bias=False
            ),
            # nn.BatchNorm2d(256),
            # nn.ReLU(True),
            # Layer 3
            nn.ConvTranspose2d(
                out_channels_list[4], out_channels_list[3], 4, 2, 1, bias=False
            ),
            # nn.BatchNorm2d(128),
            # nn.ReLU(True),
            # Layer 4
            nn.ConvTranspose2d(
                out_channels_list[3], out_channels_list[2], 4, 2, 1, bias=False
            ),
            # nn.BatchNorm2d(64),
            # nn.ReLU(True),
            # Layer 5
            nn.ConvTranspose2d(
                out_channels_list[2], out_channels_list[1], 4, 2, 1, bias=False
            ),
            # nn.Tanh()
            # Layer 6
            nn.ConvTranspose2d(out_channels_list[1], 16, 4, 2, 1, bias=False),
        )

    def forward(self, x):
        intermediate_outputs = []
        for layer in self.main:
            # print("x IN :", x.shape)
            # print("layer:", layer)
            x = layer(x)
            # print("x OUT:", x.shape)
            # print()
            intermediate_outputs.append(
                x.clone()
            )  # Clone to detach from computation graph
        # intermediate_outputs.reverse()
        return intermediate_outputs
