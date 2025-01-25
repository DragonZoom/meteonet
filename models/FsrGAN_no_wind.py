import torch
import torch.nn as nn
import torch.nn.functional as F

from .FsrGAN_helpers import (
    DownsampleBlock,
    SelfAttention,
    FirstStageEncoder,
    FirstStageDecoderBlock,
)

class RadarFirstStage(nn.Module):
    """
    Radar-only first stage:  
      - Encodes T frames of radar  
      - Decodes to T frames (or pred_len frames, if you prefer).
    """

    def __init__(self, input_len, pred_len):
        super(RadarFirstStage, self).__init__()
        self.input_len = input_len
        self.pred_len = pred_len

        # Example: 1 input channel -> 4 -> 8 -> 16
        self.encoder = FirstStageEncoder(
            T=input_len,
            in_channels=1,
            er0_channels=4,
            er1_channels=8,
            er2_channels=16
        )

        self.down_sample = nn.Sequential(
            DownsampleBlock(input_len, 4 * input_len),
            DownsampleBlock(4 * input_len, 8 * input_len),
            DownsampleBlock(8 * input_len, 16 * input_len),
        )

        self.dec3 = FirstStageDecoderBlock(in_channels=16, out_channels=8, T=input_len)
        self.dec2 = FirstStageDecoderBlock(in_channels=8, out_channels=4, T=input_len)
        self.dec1 = FirstStageDecoderBlock(in_channels=4, out_channels=2, T=input_len)

        # final conv that maps up to `pred_len` frames
        self.final_conv = nn.Conv2d(
            in_channels=2 * input_len, out_channels=pred_len, kernel_size=1
        )

    def forward(self, radar_data):
        """
        radar_data: (B, input_len, 1, H, W)
        returns:    (B, pred_len, 1, H, W)
        """
        E1, E2, E3 = self.encoder(radar_data)
        # Possibly select last hidden state or pass them forward 
        # For simplicity, let's just decode from E3 (the final level).
        # E3_repeated = E3.repeat(1, self.input_len, 1, 1, 1)  # Replicate E3 input_len times
        # out3 = self.dec3(E3)  # (B, T, 8, ...)
        B, T, _, H, W = radar_data.size()
        down_sampled = self.down_sample(radar_data.squeeze(2))
        down_sampled = down_sampled.view(B, T, 16, H // 8, W // 8)

        out3 = self.dec3(down_sampled, E3)  # (B, T, 8, ...)
        out2 = self.dec2(out3, E2)  # (B, T, 4, ...)
        out1 = self.dec1(out2, E1)  # (B, T, 2, ...)
        
        B, T, C, H, W = out1.size()
        out = out1.view(B, C*T, H, W)
        out = self.final_conv(out)
        out = out.view(B, self.pred_len, 1, H, W)
        return out
    
class RadarSecondStageGenerator(nn.Module):
    def __init__(self, input_len, pred_len, size_factor=1):
        super(RadarSecondStageGenerator, self).__init__()
        self.input_len = input_len
        self.pred_len = pred_len

        # example channels
        in_channels = input_len + pred_len  # e.g. T_in + T_pred
        base_ch = 32 // size_factor

        # ---- Encoder (simple) ----
        self.en_conv = nn.Conv2d(in_channels, base_ch, kernel_size=4, stride=2, padding=1)
        self.en2 = nn.Sequential(
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(base_ch, base_ch * 2, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(base_ch * 2),
        )
        self.en3 = nn.Sequential(
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(base_ch* 2, base_ch * 4, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(base_ch * 4),
        )
        # deeper layers, etc. – can expand as needed

        # optional self-attention
        self.att3 = SelfAttention(base_ch * 4)

        # ---- Decoder (simple) ----
        self.dc3 = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(base_ch * 4, base_ch * 2, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(base_ch * 2)
        )
        self.dc2 = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(base_ch * 2, base_ch, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(base_ch)
        )
        self.dc1 = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(base_ch, base_ch, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(base_ch)
        )
        self.final = nn.Conv2d(base_ch, pred_len, kernel_size=1)

    def forward(self, gt_radar, first_stage_pred):
        """
        gt_radar:        (B, input_len, 1, H, W)
        first_stage_pred: (B, pred_len, 1, H, W)
        returns: refined second-stage prediction of shape (B, pred_len, H, W)
        """
        B, T_in, _, H, W = gt_radar.size()
        B, T_out, _, _, _ = first_stage_pred.size()

        # flatten
        gt_radar  = gt_radar.view(B, T_in, H, W)        # (B, T_in, H, W)
        first_stage_pred = first_stage_pred.view(B, T_out, H, W) # (B, T_out, H, W)

        # concat: (B, T_in + T_out, H, W)
        x = torch.cat([gt_radar, first_stage_pred], dim=1)

        # Encoder
        e1 = self.en_conv(x)   # -> (B, base_ch, H/2, W/2)
        e2 = self.en2(e1)      # -> (B, base_ch, H/2, W/2)
        e3 = self.en3(e2)      # -> (B, base_ch, H/2, W/2)
        e3 = self.att3(e3)     # self-attention if desired

        # Decoder
        d3 = self.dc3(e3)      # upsample
        d2 = self.dc2(d3)      # upsample
        d1 = self.dc1(d2)      # upsample
        out= self.final(d1)    # -> (B, pred_len, H, W)

        # Optionally add it to the first stage pred for residual learning
        out = out + first_stage_pred
        #
        # If you do that, you’d have to reshape to match the shape of x_pred
        # out = out.unsqueeze(2) if you want shape (B, T_out, 1, H, W)
        # For demonstration, we’ll just output the raw out as the refined frames:

        # Reshape final if you want (B, T_out, 1, H, W)
        out = out.view(B, T_out, 1, H, W)
        return out
