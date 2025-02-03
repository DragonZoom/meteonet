import torch
import torch.nn as nn
import torch.nn.functional as F

from .trajGRU import TrajGRU


def separate_radar_wind(x):
    radars    = x[:,:, 0:1].contiguous()
    wind_maps = x[:,:, 1:].contiguous()
    return radars, wind_maps

###############################################################################
#                        Basic Building Blocks
###############################################################################
class DownsampleBlock(nn.Module):
    """
    Downsampling block: can be used to reduce spatial dimensions by a factor of 2.
    """

    def __init__(self, in_channels, out_channels):
        super(DownsampleBlock, self).__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size=4, stride=2, padding=1
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.conv(x))


###############################################################################
#                           First-Stage Encoders
###############################################################################
class FirstStageEncoderBlock(nn.Module):
    """
    A basic encoder block for radar data.
    """

    def __init__(self, in_channels: int, out_channels: int, T: int, h: int, w: int):
        super(FirstStageEncoderBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(
                in_channels * T,
                out_channels * T,
                kernel_size=3,
                stride=2,
                padding=1,
            ),
            nn.LeakyReLU(0.2, inplace=True),
        )
        # Build a TrajGRU that receives `out_channels` as input+output
        # The 'b_h_w' argument = (batch_size, H, W).
        # We can put a dummy batch_size=1 here; it uses H/W primarily.
        self.traj_gru = TrajGRU(
            input_channel=out_channels,
            num_filter=out_channels,
            b_h_w=(1, h, w),
            zoneout=0.0,
            L=13,
            i2h_kernel=(3, 3),
            i2h_pad=(1, 1),
            h2h_kernel=(5, 5),
        )

    def forward(self, x):
        """
        x: (B, T, in_channels, H, W)
        returns: (B, T, out_channels, H/2, W/2)
        """
        # Flatten T into chanel dimension\
        B, T, C, H, W = x.shape
        x = x.view(B, C * T, H, W)
        # Apply Conv2d + LeakyReLU
        y = self.conv(x)
        # Reshape back to (B, T, out_channels, H/2, W/2)
        y = y.view(B, T, -1, H // 2, W // 2)

        # Reorder to (T, B, C, H, W) for TrajGRU
        y = y.permute(1, 0, 2, 3, 4).contiguous()
        y, _ = self.traj_gru.forward(seq_len=T, inputs=y)
        # out_seq => (T, B, out_channels, newH, newW)
        # Reorder back to (B, T, C, H, W)
        y = y.permute(1, 0, 2, 3, 4).contiguous()
        return y


class FirstStageEncoder(nn.Module):
    """
    Radar encoder (REN) to encode radar echo sequences.
    """

    def __init__(self, T, in_channels, er0_channels, er1_channels, er2_channels):
        super(FirstStageEncoder, self).__init__()
        self.layers = nn.ModuleList(
            [
                FirstStageEncoderBlock(in_channels, er0_channels, T, h=64, w=64),
                FirstStageEncoderBlock(er0_channels, er1_channels, T, h=32, w=32),
                FirstStageEncoderBlock(er1_channels, er2_channels, T, h=16, w=16),
            ]
        )

    def forward(self, x, er2_full=False):
        """
        x: (B, T, in_channels, H, W)
        returns:
            ER0 - (B, T, 2*in_channels, H/2, W/2)
            ER1 - (B, T, 4*in_channels, H/4, W/4)
            ER2 - (B, T, 8*in_channels, H/8, W/8)
        """
        ER0 = self.layers[0](x)
        ER1 = self.layers[1](ER0)
        ER2 = self.layers[2](ER1)
        # [:,-1] - take the last hidden state
        return ER0[:, -1], ER1[:, -1], ER2 if er2_full else ER2[:, -1]


###############################################################################
#                           First-Stage Decoder
###############################################################################
class FirstStageDecoderBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, T: int, h: int, w: int):
        super(FirstStageDecoderBlock, self).__init__()
        self.out_channels = out_channels

        # Build a TrajGRU that receives `in_channels` as input+output
        # The 'b_h_w' argument = (batch_size, H, W).
        # We can put a dummy batch_size=1 here; it uses H/W primarily.
        self.traj_gru = TrajGRU(
            input_channel=in_channels,
            num_filter=in_channels,
            b_h_w=(1, h, w),
            zoneout=0.0,
            L=13,
            i2h_kernel=(3, 3),
            i2h_pad=(1, 1),
            h2h_kernel=(5, 5),
        )

        self.conv = nn.ConvTranspose2d(
            in_channels * T, out_channels * T, kernel_size=4, stride=2, padding=1
        )
        self.relu = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x, h_init=None):
        # Reorder to (T, B, C, H, W) for TrajGRU
        x = x.permute(1, 0, 2, 3, 4).contiguous()
        y, _ = self.traj_gru.forward(seq_len=x.size(0), inputs=x, states=h_init)
        # out_seq => (T, B, out_channels, newH, newW)
        # Reorder back to (B, T, C, H, W)
        y = y.permute(1, 0, 2, 3, 4).contiguous()

        # Flatten T into chanel dimension
        B, T, C, H, W = y.size()
        y = y.view(B, C * T, H, W)
        # Apply Conv2d + ReLU
        y = self.relu(self.conv(y))
        # Reshape back to (B, T, out_channels, H * 2, W * 2)
        y = y.view(B, T, self.out_channels, H * 2, W * 2)
        return y


###############################################################################
#                     Spatial and Channel Attention (SCA)
###############################################################################
class SCA(nn.Module):
    """
    Spatial-Channel Attention (SCA) block as shown in Fig.5 of the paper.

    Inputs:
      E_H, E_R: tensors of shape (B, C, H, W)
                 Must be the same shape.

    Output:
      H:        tensor of shape (B, C, H, W), computed by merging the
                spatial attention output (on E_H) and channel attention
                output (on E_R).
    """

    def __init__(self, channels):
        """
        Args:
          channels (int): number of feature channels in both E_H and E_R
        """
        super(SCA, self).__init__()

        # ----------------------------
        # 1) SPATIAL ATTENTION BRANCH
        #    Takes E_H => computes mean & max (per-channel) => concat => conv => sigmoid
        # ----------------------------
        self.conv_spatial = nn.Sequential(
            nn.Conv2d(
                in_channels=2, out_channels=1, kernel_size=7, padding=3, bias=False
            ),
            nn.BatchNorm2d(1),
            nn.Sigmoid(),
        )

        # ----------------------------
        # 2) CHANNEL ATTENTION BRANCH
        #    Takes E_R => uses both AvgPool2D & MaxPool2D => MLP => combine => sigmoid
        # ----------------------------
        # We'll define two small linear paths for avg‐pooled and max‐pooled features
        reduction = max(1, channels // 8)  # paper often uses a "channel // 8" reduction
        self.fc_avg = nn.Sequential(
            nn.Linear(channels, reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(reduction, channels, bias=False),
        )
        self.fc_max = nn.Sequential(
            nn.Linear(channels, reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(reduction, channels, bias=False),
        )

        self.sigmoid_channel = nn.Sigmoid()

    def forward(self, E_H, E_R):
        """
        E_H: (B, C, H, W) -> spatial attention path
        E_R: (B, C, H, W) -> channel attention path
        returns:
          H:  (B, C, H, W)
        """
        # ------------- Spatial Attention on E_H ------------- #
        # Compute mean and max along channel dim => (B, 1, H, W) each
        EH_mean = torch.mean(E_H, dim=1, keepdim=True)
        EH_max, _ = torch.max(E_H, dim=1, keepdim=True)

        # Concat => (B, 2, H, W)
        EH_cat = torch.cat((EH_mean, EH_max), dim=1)

        # Pass through conv -> BN -> Sigmoid => Spatial mask
        spatial_mask = self.conv_spatial(EH_cat)  # (B, 1, H, W)

        # Multiply with E_H (element‐wise)
        EH_spatial_att = E_H * spatial_mask  # (B, C, H, W)

        # ------------- Channel Attention on E_R ------------- #
        B, C, H, W = E_R.size()

        # AvgPool2D + Flatten
        ER_avg_pool = F.adaptive_avg_pool2d(E_R, (1, 1)).view(B, C)  # (B, C)
        # MaxPool2D + Flatten
        ER_max_pool = F.adaptive_max_pool2d(E_R, (1, 1)).view(B, C)  # (B, C)

        # Pass each through MLP
        avg_out = self.fc_avg(ER_avg_pool)  # (B, C)
        max_out = self.fc_max(ER_max_pool)  # (B, C)

        # Sum => sigmoid => channel mask
        channel_mask = self.sigmoid_channel(avg_out + max_out)  # (B, C)
        channel_mask = channel_mask.unsqueeze(-1).unsqueeze(-1)  # (B, C, 1, 1)

        # Multiply with E_R (element‐wise)
        ER_channel_att = E_R * channel_mask  # (B, C, H, W)

        # ------------- Merge the two results ------------- #
        # Paper’s figure shows them summed into final H
        H = EH_spatial_att + ER_channel_att  # element‐wise sum
        return H

class SSA(nn.Module):
    """
    Spatial-Spatial Attention (SSA) block as shown in Fig.5 of the paper.

    Inputs:
      E_W, E_R: tensors of shape (B, C, H, W)
                 Must be the same shape.

    Output:
      H:        tensor of shape (B, C, H, W), computed by merging the
                spatial attention output (on E_H) and spatial attention
                output (on E_R).
    """

    def __init__(self, channels):
        """
        Args:
          channels (int): number of feature channels in both E_H and E_R
        """
        super(SSA, self).__init__()

        # ----------------------------
        # 1) SPATIAL ATTENTION BRANCH for E_H
        #    Takes E_H => computes mean & max (per-channel) => concat => conv => sigmoid
        # ----------------------------
        self.conv_spatial_EH = nn.Sequential(
            nn.Conv2d(
                in_channels=2, out_channels=1, kernel_size=7, padding=3, bias=False
            ),
            nn.BatchNorm2d(1),
            nn.Sigmoid(),
        )

        # ----------------------------
        # 2) SPATIAL ATTENTION BRANCH for E_R
        #    Takes E_R => computes mean & max (per-channel) => concat => conv => sigmoid
        # ----------------------------
        self.conv_spatial_ER = nn.Sequential(
            nn.Conv2d(
                in_channels=2, out_channels=1, kernel_size=7, padding=3, bias=False
            ),
            nn.BatchNorm2d(1),
            nn.Sigmoid(),
        )

    def forward(self, E_H, E_R):
        """
        E_H: (B, C, H, W) -> spatial attention path
        E_R: (B, C, H, W) -> spatial attention path
        returns:
          H:  (B, C, H, W)
        """
        # ------------- Spatial Attention on E_H ------------- #
        # Compute mean and max along channel dim => (B, 1, H, W) each
        EH_mean = torch.mean(E_H, dim=1, keepdim=True)
        EH_max, _ = torch.max(E_H, dim=1, keepdim=True)

        # Concat => (B, 2, H, W)
        EH_cat = torch.cat((EH_mean, EH_max), dim=1)

        # Pass through conv -> BN -> Sigmoid => Spatial mask
        spatial_mask_EH = self.conv_spatial_EH(EH_cat)  # (B, 1, H, W)

        # Multiply with E_H (element‐wise)
        EH_spatial_att = E_H * spatial_mask_EH  # (B, C, H, W)

        # ------------- Spatial Attention on E_R ------------- #
        # Compute mean and max along channel dim => (B, 1, H, W) each
        ER_mean = torch.mean(E_R, dim=1, keepdim=True)
        ER_max, _ = torch.max(E_R, dim=1, keepdim=True)

        # Concat => (B, 2, H, W)
        ER_cat = torch.cat((ER_mean, ER_max), dim=1)

        # Pass through conv -> BN -> Sigmoid => Spatial mask
        spatial_mask_ER = self.conv_spatial_ER(ER_cat)  # (B, 1, H, W)

        # Multiply with E_R (element‐wise)
        ER_spatial_att = E_R * spatial_mask_ER  # (B, C, H, W)

        # ------------- Merge the two results ------------- #
        # Paper’s figure shows them summed into final H
        H = EH_spatial_att + ER_spatial_att  # element‐wise sum
        return H


###############################################################################
#                          Generator's Blocks
###############################################################################
class ResidualAttnBlock(nn.Module):
    """
    A single 'residual + attention' block using 2D convolutions.
    Inspired by the 3D variant in ED-DRAP (N-RSSAB),
    but adapted to 2D (N-2DSRAB).
    """

    def __init__(
        self, in_channels: int, expansion: int = 2, use_attention: bool = True
    ):
        super().__init__()
        self.use_attention = use_attention

        # Expand channels internally
        hidden_channels = in_channels * expansion

        # 1) Convolution to expand channels
        self.conv_expand = nn.Conv2d(
            in_channels, hidden_channels, kernel_size=1, stride=1, padding=0
        )
        self.bn_expand = nn.BatchNorm2d(hidden_channels)

        # 2) Main 3×3 Conv
        self.conv_main = nn.Conv2d(
            hidden_channels, hidden_channels, kernel_size=3, stride=1, padding=1
        )
        self.bn_main = nn.BatchNorm2d(hidden_channels)

        # 3) (Optional) Spatial / Channel Attention
        if use_attention:
            self.attn = SpatialScalingAttention(channels=hidden_channels)
        else:
            self.attn = nn.Identity()

        # 4) Compress back to original channels
        self.conv_compress = nn.Conv2d(
            hidden_channels, in_channels, kernel_size=1, stride=1, padding=0
        )
        self.bn_compress = nn.BatchNorm2d(in_channels)

        # Activation
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        # Save skip connection
        identity = x

        # Expand
        out = self.conv_expand(x)
        out = self.bn_expand(out)
        out = self.relu(out)

        # Main 3×3 conv
        out = self.conv_main(out)
        out = self.bn_main(out)
        out = self.relu(out)

        # Spatial (or channel) attention
        out = self.attn(out)

        # Compress back
        out = self.conv_compress(out)
        out = self.bn_compress(out)

        # Residual
        out = out + identity
        out = self.relu(out)
        return out


class SpatialScalingAttention(nn.Module):
    """
    Example local 'spatial scaling' or channel-spatial attention.
    Typically:
      - Might do multi-scale merges or simpler "channel+spatial" gating.
    Here, we provide a simple version that re-scales the features
    using a channel attention + a spatial gating approach.
    """

    def __init__(self, channels: int):
        super().__init__()

        # Channel attention (SE-like)
        reduction = max(1, channels // 8)
        self.mlp_avg = nn.Sequential(
            nn.Linear(channels, reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(reduction, channels, bias=False),
        )
        self.mlp_max = nn.Sequential(
            nn.Linear(channels, reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(reduction, channels, bias=False),
        )

        # Spatial attention
        self.conv_spatial = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False),
            nn.BatchNorm2d(1),
            nn.Sigmoid(),
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """
        x: (B, C, H, W)
        returns: (B, C, H, W)
        """
        B, C, H, W = x.size()
        out = x

        # ----- Channel Attention -----
        avg_pool = F.adaptive_avg_pool2d(x, 1).view(B, C)
        max_pool = F.adaptive_max_pool2d(x, 1).view(B, C)

        avg_out = self.mlp_avg(avg_pool)  # (B, C)
        max_out = self.mlp_max(max_pool)  # (B, C)
        channel_mask = self.sigmoid(avg_out + max_out).view(B, C, 1, 1)

        out = out * channel_mask

        # ----- Spatial Attention -----
        mean_map = torch.mean(out, dim=1, keepdim=True)  # (B, 1, H, W)
        max_map, _ = torch.max(out, dim=1, keepdim=True)  # (B, 1, H, W)
        spatial_in = torch.cat([mean_map, max_map], dim=1)  # (B, 2, H, W)

        spatial_mask = self.conv_spatial(spatial_in)  # (B, 1, H, W)
        out = out * spatial_mask

        return out


class N2DSRAB(nn.Module):
    """
    N-2DSRAB: N-stacked 2D Residual Spatial Attention Blocks (adapted from N-RSSAB).
    Each sub-block typically:
      - Uses 2D Convs (instead of 3D).
      - Applies a residual connection.
      - Includes a 'spatial scaling' or channel-spatial attention mechanism.
    """

    def __init__(
        self,
        in_channels: int,
        expansion: int = 2,
        num_stacks: int = 1,
        use_attention: bool = True,
    ):
        """
        Args:
          in_channels  (int): The number of input (and output) channels.
          expansion    (int): How much we expand the channels internally
                              for the hidden layers. e.g., 2, 4, 8, etc.
          num_stacks   (int): How many sub-blocks (stacks) to include.
          use_attention(bool): Whether to include spatial (or channel) attention
                               in each sub-block.
        """
        super().__init__()
        self.num_stacks = num_stacks

        # Create 'num_stacks' sub-blocks, each a ResidualAttnBlock
        blocks = []
        for _ in range(num_stacks):
            blocks.append(
                ResidualAttnBlock(
                    in_channels=in_channels,
                    expansion=expansion,
                    use_attention=use_attention,
                )
            )
        self.blocks = nn.Sequential(*blocks)

    def forward(self, x):
        # Pass input x through the stacked residual attention blocks
        return self.blocks(x)


###############################################################################
#                         Self-Attention Block
###############################################################################
class SelfAttention(nn.Module):
    """
    A typical SAGAN-style self-attention block for 2D feature maps.
    """

    def __init__(self, in_channels):
        super(SelfAttention, self).__init__()
        # Typically, we reduce dimensionality for Q/K/V.
        self.query_conv = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)

        # A learnable scale parameter
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        """
        x: (B, C, H, W)
        return: (B, C, H, W)
        """
        B, C, H, W = x.size()

        # 1) Flatten spatial dims: (B, C, H*W)
        proj_query = self.query_conv(x).view(B, -1, H * W)  # (B, C/8, H*W)
        proj_key = self.key_conv(x).view(B, -1, H * W)  # (B, C/8, H*W)
        proj_value = self.value_conv(x).view(B, -1, H * W)  # (B, C,   H*W)

        # 2) Compute attention map: (B, H*W, H*W)
        attn = torch.bmm(proj_query.permute(0, 2, 1), proj_key)  # (B, H*W, H*W)

        # 3) Normalize with softmax
        attn = F.softmax(attn, dim=-1)

        # 4) Weighted sum of the values
        out = torch.bmm(proj_value, attn.permute(0, 2, 1))  # (B, C, H*W)

        # 5) Reshape back to (B, C, H, W)
        out = out.view(B, C, H, W)

        # 6) Learnable residual scale
        out = self.gamma * out + x
        return out
