import torch
import torch.nn as nn
import torch.nn.functional as F

from .FsrGAN_helpers import (
    DownsampleBlock,
    SelfAttention,
    N2DSRAB,
    SCA,
    SSA,
    FirstStageEncoder,
    FirstStageDecoderBlock,
)

from .FsrGAN_helpers import separate_radar_wind

###############################################################################
#              First Stage: Fuse Radar & Satellite to Predict R_{t+1...t+T}
###############################################################################
class FirstStage(nn.Module):
    def __init__(self, input_len, pred_len, size_factor=1):
        super(FirstStage, self).__init__()
        self.pred_len = pred_len
        self.size_factor = size_factor

        self.ren = FirstStageEncoder(
            input_len,
            1,
            er0_channels=4 * size_factor,
            er1_channels=8 * size_factor,
            er2_channels=16 * size_factor,
        )
        self.sen = FirstStageEncoder(
            input_len,
            2,
            er0_channels=4 * size_factor,
            er1_channels=8 * size_factor,
            er2_channels=16 * size_factor,
        )

        # in origina; paper it is for 4 selected satellite channels
        self.wind_map_down_sample = nn.Sequential(
            DownsampleBlock(2 * input_len, 4 * input_len * size_factor),
            DownsampleBlock(4 * input_len * size_factor, 8 * input_len * size_factor),
            DownsampleBlock(8 * input_len * size_factor, 16 * input_len * size_factor),
        )

        self.sca_large = SSA(16 * size_factor)
        self.sca_middle = SSA(8 * size_factor)

        self.rdn3 = FirstStageDecoderBlock(16 * size_factor, 8 * size_factor, T=input_len, h=16, w=16)
        self.rdn2 = FirstStageDecoderBlock(8 * size_factor, 4 * size_factor, T=input_len, h=32, w=32)
        self.rdn1 = FirstStageDecoderBlock(4 * size_factor, 2 * size_factor, T=input_len, h=64, w=64)
        self.final_conv = nn.Conv2d(
            in_channels=2 * input_len * size_factor, out_channels=pred_len, kernel_size=1
        )

    def forward(self, x):
        """
        x: (batch_size, input_len, 3, 128, 128)
        T: int - number of time steps to predict
        """
        radar_data, wind_data = separate_radar_wind(x)
        # radar encoder
        ER0, ER1, ER2 = self.ren(radar_data)

        # wind encoder
        _, EH1, EH2 = self.sen(wind_data)

        # fusion
        H1 = self.sca_middle(EH1, ER1)
        H2 = self.sca_large(EH2, ER2)

        # wind map downsample
        B, T, C, H, W = wind_data.size()
        wind_data = wind_data.view(B, C * T, H, W)
        wind_map = self.wind_map_down_sample(wind_data)
        wind_map = wind_map.view(B, T, 16 * self.size_factor, H // 8, W // 8)

        # decoder
        out = self.rdn3(wind_map, H2)
        out = self.rdn2(out, H1)
        out = self.rdn1(out, ER0)

        # Flatten T into chanel dimension
        B, T, C, H, W = out.size()
        out = out.view(B, C * T, H, W)
        out = self.final_conv(out)
        # reshape back
        out = out.view(B, self.pred_len, 1, H, W)
        return out


###############################################################################
#                         Second-Stage Discriminator
###############################################################################
class FsrDiscriminator(nn.Module):
    """
    A patch-based or global discriminator for adversarial training.
    Here we define a small CNN that classifies real/fake.
    """
    def __init__(self, input_len, size_factor=1):
        super(FsrDiscriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(input_len, 64 // size_factor, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64 // size_factor, 128 // size_factor, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128 // size_factor),
            nn.LeakyReLU(0.2),
            nn.Conv2d(128 // size_factor, 256 // size_factor, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256 // size_factor),
            nn.LeakyReLU(0.2),
            nn.Conv2d(256 // size_factor, 512 // size_factor, kernel_size=4, stride=2, padding=0),
            nn.BatchNorm2d(512 // size_factor),
            nn.LeakyReLU(0.2),
            nn.Conv2d(512 // size_factor, 1, kernel_size=4, stride=4, padding=0),
            nn.Sigmoid(),
        )


    def forward(self, x):
        """
        x: shape (B, input_len, 1, H, W)
        returns: shape (B, input_len), real/fake classification for each time stamp
        """
        B, T, C, H, W = x.size()
        assert C == 1, "Discriminator input should have 1 channel"
        x = x.view(B, T, H, W)

        y = self.model(x)
        # flatten to (B, input_len)
        y = y.view(x.size(0), -1)

        EXPECTED = 1
        assert y.size(1) == EXPECTED, f"Expected {EXPECTED} outputs, got {y.size(1)}"
        return y


###############################################################################
#                         Second-Stage Generator
###############################################################################
class FsrSecondStageGenerator(nn.Module):
    def __init__(self, input_len, pred_len, size_factor=1):
        super(FsrSecondStageGenerator, self).__init__()
        in_channels = input_len + pred_len
        self.pred_len = pred_len
        self.input_len = input_len

        # encoder
        self.en_conv = nn.Conv2d(
            in_channels, 32 // size_factor, kernel_size=4, stride=2, padding=1
        )
        self.en2 = nn.Sequential(
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(
                32 // size_factor, 32 // size_factor, kernel_size=3, stride=1, padding=1
            ),
            nn.BatchNorm2d(32 // size_factor),
        )
        self.en3 = nn.Sequential(
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(
                32 // size_factor, 32 // size_factor, kernel_size=3, stride=1, padding=1
            ),
            nn.BatchNorm2d(32 // size_factor),
        )
        self.en4 = nn.Sequential(
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(
                32 // size_factor, 64 // size_factor, kernel_size=4, stride=2, padding=1
            ),
            nn.BatchNorm2d(64 // size_factor),
        )
        self.en5 = nn.Sequential(
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(
                64 // size_factor, 64 // size_factor, kernel_size=3, stride=1, padding=1
            ),
            nn.BatchNorm2d(64 // size_factor),
        )
        self.en6 = nn.Sequential(
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(
                64 // size_factor,
                128 // size_factor,
                kernel_size=4,
                stride=2,
                padding=1,
            ),
            nn.BatchNorm2d(128 // size_factor),
        )
        self.en7 = nn.Sequential(
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(
                128 // size_factor,
                256 // size_factor,
                kernel_size=4,
                stride=2,
                padding=1,
            ),
            nn.BatchNorm2d(256 // size_factor),
        )
        self.en8 = nn.Sequential(
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(
                256 // size_factor,
                512 // size_factor,
                kernel_size=4,
                stride=2,
                padding=1,
            ),
        )

        self.en_att5 = SelfAttention(64 // size_factor)
        self.en_att6 = SelfAttention(128 // size_factor)
        self.en_att7 = SelfAttention(256 // size_factor)
        self.en_att8 = SelfAttention(512 // size_factor)

        # fusion
        self.fs_conv = nn.Conv2d(
            input_len * 2, 32 // size_factor, kernel_size=4, stride=2, padding=1
        )
        self.fs2 = nn.Sequential(
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(
                32 // size_factor, 32 // size_factor, kernel_size=3, stride=1, padding=1
            ),
            nn.BatchNorm2d(32 // size_factor),
        )
        self.fs3 = nn.Sequential(
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(
                32 // size_factor, 32 // size_factor, kernel_size=3, stride=1, padding=1
            ),
            nn.BatchNorm2d(32 // size_factor),
        )
        self.fs4 = nn.Sequential(
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(
                32 // size_factor, 64 // size_factor, kernel_size=4, stride=2, padding=1
            ),
            nn.BatchNorm2d(64 // size_factor),
        )
        self.fs5 = nn.Sequential(
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(
                64 // size_factor, 64 // size_factor, kernel_size=3, stride=1, padding=1
            ),
            nn.BatchNorm2d(64 // size_factor),
        )
        self.fs6 = nn.Sequential(
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(
                64 // size_factor,
                128 // size_factor,
                kernel_size=4,
                stride=2,
                padding=1,
            ),
            nn.BatchNorm2d(128 // size_factor),
        )
        self.fs7 = nn.Sequential(
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(
                128 // size_factor,
                256 // size_factor,
                kernel_size=4,
                stride=2,
                padding=1,
            ),
            nn.BatchNorm2d(256 // size_factor),
        )

        self.fs_dsrab1 = N2DSRAB(32 // size_factor, expansion=1, num_stacks=1)
        self.fs_dsrab2 = N2DSRAB(32 // size_factor, expansion=1, num_stacks=2)
        self.fs_dsrab3 = N2DSRAB(32 // size_factor, expansion=1, num_stacks=2)
        self.fs_dsrab4 = N2DSRAB(64 // size_factor, expansion=1, num_stacks=4)
        self.fs_dsrab5 = N2DSRAB(64 // size_factor, expansion=1, num_stacks=4)
        self.fs_dsrab6 = N2DSRAB(128 // size_factor, expansion=1, num_stacks=8)
        self.fs_dsrab7 = N2DSRAB(256 // size_factor, expansion=1, num_stacks=8)

        # decoder
        self.dc8 = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(
                512 // size_factor,
                out_channels=256 // size_factor,
                kernel_size=4,
                stride=2,
                padding=1,
            ),
            nn.BatchNorm2d(256 // size_factor),
            nn.Dropout2d(inplace=True),
        )
        self.dc7 = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(
                3 * 256 // size_factor,
                out_channels=128 // size_factor,
                kernel_size=4,
                stride=2,
                padding=1,
            ),
            nn.BatchNorm2d(128 // size_factor),
            nn.Dropout2d(inplace=True),
        )
        self.dc6 = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(
                3 * 128 // size_factor,
                out_channels=64 // size_factor,
                kernel_size=4,
                stride=2,
                padding=1,
            ),
            nn.BatchNorm2d(64 // size_factor),
            nn.Dropout2d(inplace=True),
        )
        #
        self.dc5 = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(
                3 * 64 // size_factor,
                out_channels=64 // size_factor,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.BatchNorm2d(64 // size_factor),
        )
        self.dc5_dsrab = N2DSRAB(
            in_channels=64 // size_factor, expansion=4, num_stacks=4
        )
        #
        self.dc4 = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(
                3 * 64 // size_factor,
                out_channels=32 // size_factor,
                kernel_size=4,
                stride=2,
                padding=1,
            ),
            nn.BatchNorm2d(32 // size_factor),
        )
        self.dc4_dsrab = N2DSRAB(
            in_channels=32 // size_factor, expansion=2, num_stacks=2
        )
        #
        self.dc3 = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(
                3 * 32 // size_factor,
                out_channels=32 // size_factor,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.BatchNorm2d(32 // size_factor),
        )
        self.dc3_dsrab = N2DSRAB(
            in_channels=32 // size_factor, expansion=1, num_stacks=1
        )
        #
        self.dc2 = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(
                3 * 32 // size_factor,
                out_channels=32 // size_factor,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.BatchNorm2d(32 // size_factor),
        )
        self.dc1 = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(
                3 * 32 // size_factor,
                out_channels=pred_len,
                kernel_size=4,
                stride=2,
                padding=1,
            ),
        )

    def forward(self, gt_x, first_stage_pred):
        gt_radar, gt_wind = separate_radar_wind(gt_x)
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
        enc2 = self.en2(enc1)
        enc3 = self.en3(enc2)
        enc4 = self.en4(enc3)
        enc5 = self.en5(enc4)
        enc6 = self.en6(enc5)
        enc7 = self.en7(enc6)
        enc8 = self.en8(enc7)
        # attention
        enc5 = self.en_att5(enc5)
        enc6 = self.en_att6(enc6)
        enc7 = self.en_att7(enc7)
        enc8 = self.en_att8(enc8)

        # Fusion
        fs1 = self.fs_conv(gt_wind)
        fs2 = self.fs2(fs1)
        fs3 = self.fs3(fs2)
        fs4 = self.fs4(fs3)
        fs5 = self.fs5(fs4)
        fs6 = self.fs6(fs5)
        fs7 = self.fs7(fs6)
        # DSRAB
        # fs1 = self.fs_dsrab1(fs1)
        # fs2 = self.fs_dsrab2(fs2)
        # fs3 = self.fs_dsrab3(fs3)
        # fs4 = self.fs_dsrab4(fs4)
        # fs5 = self.fs_dsrab5(fs5)
        # fs6 = self.fs_dsrab6(fs6)
        # fs7 = self.fs_dsrab7(fs7)

        # Decoder
        dec8 = self.dc8(enc8)
        dec7 = self.dc7(torch.cat([enc7, dec8, fs7], dim=1))
        dec6 = self.dc6(torch.cat([enc6, dec7, fs6], dim=1))
        #
        dec5 = self.dc5(torch.cat([enc5, dec6, fs5], dim=1))
        # dec5 = self.dc5_dsrab(dec5) + dec5
        #
        dec4 = self.dc4(torch.cat([enc4, dec5, fs4], dim=1))
        # dec4 = self.dc4_dsrab(dec4) + dec4
        #
        dec3 = self.dc3(torch.cat([enc3, dec4, fs3], dim=1))
        # dec3 = self.dc3_dsrab(dec3) + dec3
        #
        dec2 = self.dc2(torch.cat([enc2, dec3, fs2], dim=1))
        dec1 = self.dc1(torch.cat([enc1, dec2, fs1], dim=1))

        # return dec1
        out = dec1 + first_stage_pred
        return out


###############################################################################
#                          Full FsrGAN Model Wrapper
###############################################################################
class FsrGAN(nn.Module):
    def __init__(self, input_len, pred_len):
        super(FsrGAN, self).__init__()
        self.first_stage = FirstStage(input_len, pred_len)
        self.second_stage = FsrSecondStageGenerator(input_len, pred_len)

    def forward(self, radar_data, wind_data):
        first_stage_pred = self.first_stage(radar_data, wind_data)
        second_stage_pred = self.second_stage(radar_data, wind_data, first_stage_pred)
        return second_stage_pred
