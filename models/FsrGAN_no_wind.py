import torch
import torch.nn as nn
import torch.nn.functional as F

from .FsrGAN_helpers import (
    DownsampleBlock,
    SelfAttention,
    FirstStageEncoder,
    FirstStageDecoderBlock,
)

from .trajGRU import TrajGRU


class RadarEncoderBlock(nn.Module):
    """
    Downsamples + processes a radar sequence with TrajGRU.
    """

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 input_len: int,
                 h: int,
                 w: int):
        """
        Args:
            in_channels: channels in the radar data (e.g. 1)
            out_channels: hidden channels after conv/downsample
            input_len: sequence length
            h, w: height & width after downsampling (for TrajGRU).
                  If you start with, e.g., 128x128 and stride=2 here,
                  then h,w = 64,64 for the TrajGRU step, etc.
        """
        super().__init__()
        self.out_channels = out_channels
        self.input_len = input_len
        self.down = nn.Conv2d(
            in_channels * input_len,
            out_channels * input_len,
            kernel_size=4,
            stride=2,
            padding=1
        )
        self.act = nn.LeakyReLU(0.2, inplace=True)
        # Build a TrajGRU that receives `out_channels` as input+output
        # The 'b_h_w' argument = (batch_size, H, W).
        # We can put a dummy batch_size=1 here; it uses H/W primarily.
        self.traj_gru = TrajGRU(
            input_channel=out_channels,
            num_filter=out_channels,
            b_h_w=(1, h, w),
            zoneout=0.0,
            L=5,  # or any # of flows
            i2h_kernel=(3,3),
            i2h_pad=(1,1),
            h2h_kernel=(5,5),
        )

    def forward(self, x):
        """
        x shape: (B, T, in_channels, H, W)
        returns: (B, T, out_channels, H/2, W/2)
        """
        B, T, C, H, W = x.shape
        # Flatten time into channels so we can downsample:
        # shape => (B, T*C, H, W)
        x = x.view(B, C*T, H, W)
        x = self.down(x)  # stride=2
        x = self.act(x)

        # Un-flatten back into time dimension:
        # shape => (B, T, out_channels, H/2, W/2)
        newH, newW = H // 2, W // 2
        x = x.view(B, T, self.out_channels, newH, newW)

        # Reorder to (T, B, C, H, W) for TrajGRU
        x = x.permute(1, 0, 2, 3, 4).contiguous()

        # Run TrajGRU over time
        # The forward method wants (seq_len=T, inputs shape = (T,B,C,H,W))
        out_seq, _ = self.traj_gru.forward(seq_len=T, inputs=x)
        # out_seq => (T, B, out_channels, newH, newW)

        # Reorder back => (B, T, out_channels, newH, newW)
        out = out_seq.permute(1, 0, 2, 3, 4).contiguous()
        return out


class RadarEncoder(nn.Module):
    """
    Simple multi-level encoder using TrajGRU at each downsampling step.
    """

    def __init__(self, input_len, in_channels):
        super().__init__()
        self.block1 = RadarEncoderBlock(
            in_channels=in_channels,
            out_channels=4,
            input_len=input_len,
            h=64, w=64  # if your input is 128x128, after stride=2 => 64x64
        )
        self.block2 = RadarEncoderBlock(
            in_channels=4,
            out_channels=8,
            input_len=input_len,
            h=32, w=32  # another stride=2 => 32x32
        )
        self.block3 = RadarEncoderBlock(
            in_channels=8,
            out_channels=16,
            input_len=input_len,
            h=16, w=16  # another stride=2 => 16x16
        )

    def forward(self, x):
        """
        x: (B, T, in_channels, 128, 128) -> example
        returns: E1, E2, E3
           E1: (B, T, 4, 64, 64)
           E2: (B, T, 8, 32, 32)
           E3: (B, T, 16,16, 16)
        """
        E1 = self.block1(x)  # down to 64x64
        E2 = self.block2(E1) # down to 32x32
        E3 = self.block3(E2) # down to 16x16
        return E1, E2, E3


###############################################################################
# RadarDecoder
###############################################################################
class RadarDecoderBlock(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 input_len: int,
                 h: int,
                 w: int):
        """
        If the incoming feature map is shape (B, T, in_channels, H, W),
        we upsample to (B, T, out_channels, 2H, 2W), then run TrajGRU again.
        """
        super().__init__()
        self.out_channels = out_channels
        self.input_len = input_len

        # Transposed conv to upsample
        self.up = nn.ConvTranspose2d(
            in_channels * input_len,
            out_channels * input_len,
            kernel_size=4,
            stride=2,
            padding=1
        )
        self.act = nn.LeakyReLU(0.2, inplace=True)

        # Another TrajGRU
        # b_h_w => after upsampling, the new resolution is 2h x 2w
        self.traj_gru = TrajGRU(
            input_channel=out_channels,
            num_filter=out_channels,
            b_h_w=(1, 2*h, 2*w),
            L=5
        )

        self.h = h
        self.w = w

    def forward(self, x):
        """
        x: (B, T, in_channels, h, w)
        returns: (B, T, out_channels, 2h, 2w)
        """
        B, T, C, h, w = x.size()
        # Flatten T into channels
        x = x.view(B, C*T, h, w)
        x = self.up(x)
        x = self.act(x)

        # Now shape => (B, out_channels*T, 2h, 2w)
        newH, newW = 2*h, 2*w
        x = x.view(B, T, self.out_channels, newH, newW)

        # run TrajGRU
        x = x.permute(1,0,2,3,4)
        out_seq, _ = self.traj_gru.forward(seq_len=T, inputs=x)
        out = out_seq.permute(1,0,2,3,4).contiguous()
        return out


###############################################################################
# RadarFirstStage
###############################################################################
class RadarFirstStage(nn.Module):
    """
    A first-stage generator that predicts future radar frames using
    TrajGRU-based encoder-decoder.
    """

    def __init__(self, input_len, pred_len, in_channels=1):
        super().__init__()
        self.pred_len = pred_len
        self.input_len = input_len
        self.encoder = RadarEncoder(input_len, in_channels)

        # 3-level decoder
        self.dec3 = RadarDecoderBlock(in_channels=16, out_channels=8, input_len=input_len, h=16, w=16)
        self.dec2 = RadarDecoderBlock(in_channels=8, out_channels=4, input_len=input_len, h=32, w=32)
        self.dec1 = RadarDecoderBlock(in_channels=4, out_channels=2, input_len=input_len, h=64, w=64)

        # final 1x1 conv to produce pred_len frames from (2 * input_len) channels
        self.final_conv = nn.Conv2d(
            in_channels=2 * input_len,
            out_channels=pred_len,  # produce pred_len frames
            kernel_size=1
        )

    def forward(self, x):
        """
        x: (B, input_len, in_channels, 128, 128)
        returns: (B, pred_len, 1, 128, 128)
        """
        E1, E2, E3 = self.encoder(x)  # encode

        # decode
        D3 = self.dec3(E3)  # => (B, T, 8, 32, 32)
        D2 = self.dec2(D3)  # => (B, T, 4, 64, 64)
        D1 = self.dec1(D2)  # => (B, T, 2, 128,128)

        # Flatten T into channels => final conv => get pred_len channels
        B, T, C, H, W = D1.shape
        out = D1.view(B, C*T, H, W)
        out = self.final_conv(out)

        # reshape => (B, pred_len, 1, 128, 128)
        out = out.view(B, self.pred_len, 1, H, W)
        return out


###############################################################################
# RadarFirstStage
###############################################################################
class RadarSecondStageGenerator(nn.Module):
    """
    Takes the original radar input and the first-stage coarse prediction
    and refines it. We'll keep it simpler (maybe just 1-2 conv layers + TrajGRU).
    """

    def __init__(self, input_len, pred_len):
        super().__init__()
        self.input_len = input_len
        self.pred_len = pred_len

        # Simple example: a single TrajGRU layer that sees (T_in + T_out) frames as input channels
        # plus a final conv that outputs T_out refined frames.

        in_ch = input_len + pred_len
        hidden = 8  # arbitrary
        self.conv_in = nn.Conv2d(
            in_ch, hidden, kernel_size=3, stride=1, padding=1
        )
        self.traj_gru = TrajGRU(
            input_channel=hidden,
            num_filter=hidden,
            b_h_w=(1, 128, 128),  # full resolution if we do no downsampling
            L=5
        )
        self.conv_out = nn.Conv2d(hidden, pred_len, kernel_size=1)

    def forward(self, radar_data, first_stage_pred):
        """
        radar_data:      (B, input_len, 1, H, W)
        first_stage_pred:(B, pred_len, 1, H, W)
        returns refined frames => (B, pred_len, 1, H, W)
        """
        B, T_in, _, H, W = radar_data.shape
        B, T_out, _, _, _ = first_stage_pred.shape

        # Flatten them along channel dimension:
        x_in  = radar_data.view(B, T_in, H, W)          # => (B, T_in, H, W)
        x_pred= first_stage_pred.view(B, T_out, H, W)   # => (B, T_out, H, W)
        x = torch.cat([x_in, x_pred], dim=1)            # => (B, T_in + T_out, H, W)

        x = self.conv_in(x)                             # => (B, hidden, H, W)

        # Reorder for TrajGRU
        x = x.unsqueeze(1)  # Insert time dimension T=1? 
                            # Actually if we want to treat each frame as time steps, 
                            # we need a different approach. 
                            # Here's a simpler approach: treat the entire stack as 
                            # "channels" at once. 
                            # We'll do T=1 to keep the code simple:
        x = x.permute(1, 0, 2, 3, 4).contiguous()       # (1, B, hidden, H, W)
        out_seq, _ = self.traj_gru.forward(seq_len=1, inputs=x)
        out = out_seq.permute(1, 0, 2, 3, 4)            # => (B, 1, hidden, H, W)
        out = out.squeeze(1)                            # => (B, hidden, H, W)

        out = self.conv_out(out)                        # => (B, pred_len, H, W)
        out = out.view(B, T_out, 1, H, W)
        return out
