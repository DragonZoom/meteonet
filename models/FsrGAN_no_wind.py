import torch
import torch.nn as nn
import torch.nn.functional as F

from .FsrGAN_helpers import (
    DownsampleBlock,
    SelfAttention,
    FirstStageEncoder,
    FirstStageDecoderBlock,
    N2DSRAB,
)

class SA(nn.Module):
    """
    Spatial Attention (SA) block as shown in Fig.5 of the paper.

    Inputs:
      E: tensor of shape (B, C, H, W)

    Output:
      H: tensor of shape (B, C, H, W), computed by applying spatial attention on E.
    """

    def __init__(self):
        """
        Args:
          channels (int): number of feature channels in E
        """
        super(SA, self).__init__()

        # ----------------------------
        # SPATIAL ATTENTION BRANCH for E
        # Takes E => computes mean & max (per-channel) => concat => conv => sigmoid
        # ----------------------------
        self.conv_spatial = nn.Sequential(
            nn.Conv2d(
                in_channels=2, out_channels=1, kernel_size=7, padding=3, bias=False
            ),
            nn.BatchNorm2d(1),
            nn.Sigmoid(),
        )

    def forward(self, E):
        """
        E: (B, C, H, W) -> spatial attention path
        returns:
          H: (B, C, H, W)
        """
        # ------------- Spatial Attention on E ------------- #
        # Compute mean and max along channel dim => (B, 1, H, W) each
        E_mean = torch.mean(E, dim=1, keepdim=True)
        E_max, _ = torch.max(E, dim=1, keepdim=True)

        # Concat => (B, 2, H, W)
        E_cat = torch.cat((E_mean, E_max), dim=1)

        # Pass through conv -> BN -> Sigmoid => Spatial mask
        spatial_mask = self.conv_spatial(E_cat)  # (B, 1, H, W)

        # Multiply with E (element‐wise)
        E_spatial_att = E * spatial_mask  # (B, C, H, W)

        return E_spatial_att


###############################################################################
#              First Stage: Radar only
###############################################################################
class FirstStageRadarOnly(nn.Module):
    def __init__(self, input_len, pred_len, size_factor=1):
        super(FirstStageRadarOnly, self).__init__()
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
            er0_channels=8 * size_factor,
            er1_channels=8 * size_factor,
            er2_channels=16 * size_factor,
        )

        # in origina; paper it is for 4 selected satellite channels
        self.rain_map_down_sample = nn.Sequential(
            DownsampleBlock(input_len, 4 * input_len * size_factor),
            DownsampleBlock(4 * input_len * size_factor, 8 * input_len * size_factor),
            DownsampleBlock(8 * input_len * size_factor, 16 * input_len * size_factor),
        )

        self.sca_large = SA()
        self.sca_middle = SA()

        self.rdn3 = FirstStageDecoderBlock(16 * size_factor, 8 * size_factor, T=input_len, h=16, w=16)
        self.rdn2 = FirstStageDecoderBlock(8 * size_factor, 4 * size_factor, T=input_len, h=32, w=32)
        self.rdn1 = FirstStageDecoderBlock(4 * size_factor, 2 * size_factor, T=input_len, h=64, w=64)
        self.final_conv = nn.Conv2d(
            in_channels=2 * input_len * size_factor, out_channels=pred_len, kernel_size=1
        )

    def forward(self, radar_data):
        """
        x: (batch_size, input_len, 1, 128, 128)
        T: int - number of time steps to predict
        """
        assert radar_data.size(2) == 1
        # radar encoder
        ER0, ER1, ER2 = self.ren(radar_data)

        # fusion
        H1 = self.sca_middle(ER1)
        H2 = self.sca_large(ER2)

        # rain map downsample
        # (Replacement of selected satellite channels with rain map)
        B, T, C, H, W = radar_data.size()
        radar_data = radar_data.view(B, C * T, H, W)
        radar_data = self.rain_map_down_sample(radar_data)
        radar_data = radar_data.view(B, T, 16 * self.size_factor, H // 8, W // 8)

        # decoder
        out = self.rdn3(radar_data, H2)
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
#                         Second-Stage Generator (Radar Only)
###############################################################################
class FsrSecondStageGeneratorRadarOnly(nn.Module):
    def __init__(self, input_len, pred_len, size_factor=1, predict_sequence=False):
        super(FsrSecondStageGeneratorRadarOnly, self).__init__()
        in_channels = input_len + pred_len
        self.pred_len = pred_len
        self.input_len = input_len
        self.predict_sequence = predict_sequence

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
                2 * 256 // size_factor,
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
                2 * 128 // size_factor,
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
                2 * 64 // size_factor,
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
                2 * 64 // size_factor,
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
                2 * 32 // size_factor,
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
                2 * 32 // size_factor,
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
                2 * 32 // size_factor,
                out_channels=pred_len if predict_sequence else 1,
                kernel_size=4,
                stride=2,
                padding=1,
            ),
        )

    def forward(self, gt_radar, first_stage_pred):
        assert gt_radar.size(2) == 1
    
        # reduce 5D to 4D
        B, _, _, H, W = gt_radar.size()
        gt_radar = gt_radar.view(B, self.input_len, H, W)

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

        # Decoder
        dec8 = self.dc8(enc8)
        dec7 = self.dc7(torch.cat([enc7, dec8], dim=1))
        dec6 = self.dc6(torch.cat([enc6, dec7], dim=1))
        #
        dec5 = self.dc5(torch.cat([enc5, dec6], dim=1))
        # dec5 = self.dc5_dsrab(dec5) + dec5
        #
        dec4 = self.dc4(torch.cat([enc4, dec5], dim=1))
        # dec4 = self.dc4_dsrab(dec4) + dec4
        #
        dec3 = self.dc3(torch.cat([enc3, dec4], dim=1))
        # dec3 = self.dc3_dsrab(dec3) + dec3
        #
        dec2 = self.dc2(torch.cat([enc2, dec3], dim=1))
        dec1 = self.dc1(torch.cat([enc1, dec2], dim=1))

        # return dec1
        if self.predict_sequence:
            out = dec1 + first_stage_pred
        else:
            out = dec1 + first_stage_pred[:, -1:]
        return out