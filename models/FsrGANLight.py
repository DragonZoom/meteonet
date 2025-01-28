import torch
import torch.nn as nn

from .FsrGAN_helpers import (
    SelfAttention,
    N2DSRAB,
)


###############################################################################
#                          Second-Stage Decoder Light
###############################################################################
class FsrSecondStageGeneratorLight(nn.Module):
    def __init__(self, input_len, pred_len):
        super(FsrSecondStageGeneratorLight, self).__init__()
        in_channels = input_len + pred_len
        self.pred_len = pred_len
        self.input_len = input_len
        
        # chs = [32, 32, 64, 128, 256]

        # encoder
        self.en_conv = nn.Conv2d(in_channels, 32, kernel_size=4, stride=2, padding=1)
        self.en3 = nn.Sequential(
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
        )
        self.en5 = nn.Sequential(
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
        )
        self.en7 = nn.Sequential(
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
        )
        self.en8 = nn.Sequential(
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
        )

        self.en_att5 = SelfAttention(64)
        self.en_att7 = SelfAttention(128)
        self.en_att8 = SelfAttention(256)

        # fusion
        self.fs_conv = nn.Conv2d(input_len * 2, 32, kernel_size=4, stride=2, padding=1)
        self.fs3 = nn.Sequential(
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
        )
        self.fs5 = nn.Sequential(
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
        )
        self.fs7 = nn.Sequential(
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
        )

        self.fs_dsrab1 = N2DSRAB(32, expansion=1, num_stacks=1)
        self.fs_dsrab3 = N2DSRAB(32, expansion=1, num_stacks=2)
        self.fs_dsrab5 = N2DSRAB(64, expansion=1, num_stacks=4)
        self.fs_dsrab7 = N2DSRAB(128, expansion=1, num_stacks=8)


        # decoder
        self.dc8 = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, out_channels=128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.Dropout2d(inplace=True)
        )
        self.dc7 = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(3*128, out_channels=64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.Dropout2d(inplace=True)
        )
        #
        self.dc5 = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(3*64, out_channels=32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32)
        )
        self.dc5_dsrab = N2DSRAB(in_channels=32, expansion=4, num_stacks=4)
        #
        self.dc3 = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(3*32, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32)
        )
        self.dc3_dsrab = N2DSRAB(in_channels=32, expansion=1, num_stacks=1)
        #
        self.dc1 = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(3*32, out_channels=pred_len, kernel_size=4, stride=2, padding=1)
        )


    def forward(self, gt_radar, gt_wind, first_stage_pred):
        # reduce 5D to 4D
        B, _, _, H, W = gt_radar.size()
        gt_radar = gt_radar.view(B, self.input_len, H, W)

        B, _, _, H, W = gt_wind.size()
        gt_wind = gt_wind.view(B, 2 * self.input_len, H, W)

        B, _, _, H, W = first_stage_pred.size()
        first_stage_pred = first_stage_pred.view(B, self.pred_len, H, W)

        # Encoder
        enc_in = torch.cat([gt_radar, first_stage_pred], dim=1)
        enc1 = self.en_conv(enc_in)
        enc3 = self.en3(enc1)
        enc5 = self.en5(enc3)
        enc7 = self.en7(enc5)
        enc8 = self.en8(enc7)
        # attention
        enc5 = self.en_att5(enc5)
        enc7 = self.en_att7(enc7)
        enc8 = self.en_att8(enc8)

        # Fusion
        fs1 = self.fs_conv(gt_wind)
        fs3 = self.fs3(fs1)
        fs5 = self.fs5(fs3)
        fs7 = self.fs7(fs5)
        # DSRAB
        fs1 = self.fs_dsrab1(fs1)
        fs3 = self.fs_dsrab3(fs3)
        fs5 = self.fs_dsrab5(fs5)
        fs7 = self.fs_dsrab7(fs7)

        # Decoder
        dec8 = self.dc8(enc8)
        dec7 = self.dc7(torch.cat([enc7, dec8, fs7], dim=1))
        #
        dec5 = self.dc5(torch.cat([enc5, dec7, fs5], dim=1))
        dec5 = self.dc5_dsrab(dec5) + dec5
        #
        dec3 = self.dc3(torch.cat([enc3, dec5, fs3], dim=1))
        dec3 = self.dc3_dsrab(dec3) + dec3
        #
        dec1 = self.dc1(torch.cat([enc1, dec3, fs1], dim=1))

        out = dec1 + first_stage_pred
        return out