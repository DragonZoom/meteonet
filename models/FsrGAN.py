import torch
import torch.nn as nn
import torch.nn.functional as F


###############################################################################
#                        TrajGRU Cell Implementation
###############################################################################
class TrajGRUCell(nn.Module):
    """
    A single TrajGRU cell. This cell aims to learn dynamic receptive fields
    and track 'trajectories' in the hidden state for spatiotemporal data.

    Reference:
    Shi et al. "Deep Learning for Precipitation Nowcasting: A Benchmark and A New Model", NIPS 2017.
    """

    def __init__(
        self, input_channels, hidden_channels, kernel_size=3, stride=1, padding=1
    ):
        super(TrajGRUCell, self).__init__()
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels

        # Gating and candidate transformations
        self.conv_zr = nn.Conv2d(
            in_channels=input_channels + hidden_channels,
            out_channels=2 * hidden_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        )

        self.conv_h = nn.Conv2d(
            in_channels=input_channels + hidden_channels,
            out_channels=hidden_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        )

        # If you implement dynamic filters or flow-based gating,
        # you'll need additional convolution layers or a separate module
        # to learn offsets/flows. This minimal version does not include them.

    def forward(self, x, h_prev):
        """
        x:      (B, input_channels,  H, W)
        h_prev: (B, hidden_channels, H, W)
        return: h: updated hidden state
        """
        if h_prev is None:
            # Initialize hidden if needed
            h_prev = torch.zeros(
                x.size(0),
                self.hidden_channels,
                x.size(2),
                x.size(3),
                device=x.device,
                dtype=x.dtype,
            )

        # Concatenate along channel dimension
        combined = torch.cat([x, h_prev], dim=1)  # (B, input+hidden, H, W)

        # 1) Gating
        zr = self.conv_zr(combined)  # (B, 2*hidden_channels, H, W)
        z, r = torch.split(zr, self.hidden_channels, dim=1)
        z = torch.sigmoid(z)
        r = torch.sigmoid(r)

        # 2) Candidate hidden
        combined_hr = torch.cat([x, r * h_prev], dim=1)  # (B, input+hidden, H, W)
        h_tilde = torch.tanh(self.conv_h(combined_hr))  # (B, hidden_channels, H, W)

        # 3) New hidden
        h = (1 - z) * h_prev + z * h_tilde
        return h


class TrajGRU(nn.Module):
    """
    A multi-step TrajGRU that processes T input frames
    to produce T hidden states (or a final one).
    """

    def __init__(
        self, input_channels, hidden_channels, kernel_size=3, stride=1, padding=1
    ):
        super(TrajGRU, self).__init__()
        self.cell = TrajGRUCell(
            input_channels, hidden_channels, kernel_size, stride, padding
        )

    def forward(self, x_seq, h_init=None):
        """
        x_seq: (B, T, input_channels, H, W)
        h_init: initial hidden state, if not provided, it's initialized to zeros (B, hidden_channels, H, W)
        returns:
          h_seq: (B, T, hidden_channels, H, W)
        """
        B, T, C, H, W = x_seq.size()
        h_seq = []
        h_prev = h_init

        for t in range(T):
            x_t = x_seq[:, t]  # (B, input_channels, H, W)
            h_prev = self.cell(x_t, h_prev)
            h_seq.append(h_prev)

        # stack across time dimension
        h_seq = torch.stack(h_seq, dim=1)  # (B, T, hidden_channels, H, W)
        return h_seq


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

    def __init__(self, in_channels, out_channels, T):
        super(FirstStageEncoderBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(
                in_channels * T,
                out_channels * T,
                kernel_size=3,
                stride=2,
                padding=1,
                bias=False,
            ),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.traj = TrajGRU(out_channels, out_channels)

    def forward(self, x):
        """
        x: (B, T, in_channels, H, W)
        returns: (B, T, out_channels, H/2, W/2)
        """
        # Flatten T into chanel dimension\
        B, T, C, H, W = x.size()
        x = x.view(B, C * T, H, W)
        # Apply Conv2d + LeakyReLU
        y = self.conv(x)
        # Reshape back to (B, T, out_channels, H/2, W/2)
        y = y.view(B, T, -1, H // 2, W // 2)
        # Apply TrajGRU
        return self.traj(y)


class FirstStageEncoder(nn.Module):
    """
    Radar encoder (REN) to encode radar echo sequences.
    """

    def __init__(self, T, in_channels, er0_channels, er1_channels, er2_channels):
        super(FirstStageEncoder, self).__init__()
        self.layers = nn.ModuleList(
            [
                FirstStageEncoderBlock(in_channels, er0_channels, T),
                FirstStageEncoderBlock(er0_channels, er1_channels, T),
                FirstStageEncoderBlock(er1_channels, er2_channels, T),
            ]
        )

    def forward(self, x):
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
        return ER0[:, -1], ER1[:, -1], ER2[:, -1]


###############################################################################
#                           First-Stage Decoder
###############################################################################
class FirstStageDecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, T):
        super(FirstStageDecoderBlock, self).__init__()
        self.out_channels = out_channels

        self.traj = TrajGRU(in_channels, in_channels)
        self.conv = nn.ConvTranspose2d(
            in_channels * T, out_channels * T, kernel_size=4, stride=2, padding=1
        )
        self.relu = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x, h_init):
        y = self.traj(x, h_init)
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


###############################################################################
#              First Stage: Fuse Radar & Satellite to Predict R_{t+1...t+T}
###############################################################################
class FirstStage(nn.Module):
    def __init__(self, input_len, pred_len):
        super(FirstStage, self).__init__()
        self.pred_len = pred_len

        self.ren = FirstStageEncoder(
            input_len,
            1,
            er0_channels=4,
            er1_channels=8,
            er2_channels=16,
        )
        self.sen = FirstStageEncoder(
            input_len,
            2,
            er0_channels=4,
            er1_channels=8,
            er2_channels=16,
        )

        # in origina; paper it is for 4 selected satellite channels
        self.wind_map_down_sample = nn.Sequential(
            DownsampleBlock(2 * input_len, 4 * input_len),
            DownsampleBlock(4 * input_len, 8 * input_len),
            DownsampleBlock(8 * input_len, 16 * input_len),
        )

        self.sca_large = SCA(16)
        self.sca_middle = SCA(8)

        self.rdn3 = FirstStageDecoderBlock(16, 8, T=input_len)
        self.rdn2 = FirstStageDecoderBlock(8, 4, T=input_len)
        self.rdn1 = FirstStageDecoderBlock(4, 2, T=input_len)
        self.final_conv = nn.Conv2d(
            in_channels=2 * input_len, out_channels=pred_len, kernel_size=1
        )

    def forward(self, radar_data, wind_data):
        """
        radar_data: (batch_size, input_len, 1, 128, 128)
        wind_data : (batch_size, input_len, 2, 128, 128)
        T: int - number of time steps to predict
        """
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
        wind_map = wind_map.view(B, T, 16, H // 8, W // 8)

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

    def __init__(self, input_len):
        super(FsrDiscriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(input_len, 32, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=0),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, input_len, kernel_size=4, stride=4, padding=0),
        )

    def forward(self, x):
        """
        x: shape (B, input_len, 1, H, W)
        returns: shape (B. input_len), real/fake classification for each time stamp
        """
        B, T, C, H, W = x.size()
        assert C == 1, "Discriminator input should have 1 channel"
        x = x.view(B, T, H, W)

        y = self.model(x)
        # flatten to (B, input_len)
        y = y.view(x.size(0), -1)
        # apply sigmoid
        y = torch.sigmoid(y)
        return y


###############################################################################
#                          Generator's Blocks
###############################################################################
class ResidualAttnBlock(nn.Module):
    """
    A single 'residual + attention' block using 2D convolutions.
    Inspired by the 3D variant in ED-DRAP (N-RSSAB),
    but adapted to 2D (N-2DSRAB).
    """
    def __init__(self, in_channels: int, expansion: int = 2, use_attention: bool = True):
        super().__init__()
        self.use_attention = use_attention

        # Expand channels internally
        hidden_channels = in_channels * expansion

        # 1) Convolution to expand channels
        self.conv_expand = nn.Conv2d(in_channels, hidden_channels, kernel_size=1, stride=1, padding=0)
        self.bn_expand   = nn.BatchNorm2d(hidden_channels)

        # 2) Main 3×3 Conv
        self.conv_main = nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, stride=1, padding=1)
        self.bn_main   = nn.BatchNorm2d(hidden_channels)

        # 3) (Optional) Spatial / Channel Attention
        if use_attention:
            self.attn = SpatialScalingAttention(channels=hidden_channels)
        else:
            self.attn = nn.Identity()

        # 4) Compress back to original channels
        self.conv_compress = nn.Conv2d(hidden_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.bn_compress   = nn.BatchNorm2d(in_channels)

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
            nn.Linear(reduction, channels, bias=False)
        )
        self.mlp_max = nn.Sequential(
            nn.Linear(channels, reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(reduction, channels, bias=False)
        )

        # Spatial attention
        self.conv_spatial = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
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
        max_map, _ = torch.max(out, dim=1, keepdim=True) # (B, 1, H, W)
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
            blocks.append(ResidualAttnBlock(
                in_channels=in_channels, 
                expansion=expansion,
                use_attention=use_attention
            ))
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
        self.key_conv   = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels, in_channels,       kernel_size=1)

        # A learnable scale parameter
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        """
        x: (B, C, H, W)
        return: (B, C, H, W)
        """
        B, C, H, W = x.size()

        # 1) Flatten spatial dims: (B, C, H*W)
        proj_query = self.query_conv(x).view(B, -1, H*W)  # (B, C/8, H*W)
        proj_key   = self.key_conv(x).view(B, -1, H*W)    # (B, C/8, H*W)
        proj_value = self.value_conv(x).view(B, -1, H*W)  # (B, C,   H*W)

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


###############################################################################
#                         Second-Stage Generator
###############################################################################
class FsrSecondStageGenerator(nn.Module):
    def __init__(self, input_len, pred_len):
        super(FsrSecondStageGenerator, self).__init__()
        in_channels = input_len + pred_len
        self.pred_len = pred_len
        self.input_len = input_len

        # encoder
        self.en_conv = nn.Conv2d(in_channels, 32, kernel_size=4, stride=2, padding=1)
        self.en2 = nn.Sequential(
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
        )
        self.en3 = nn.Sequential(
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
        )
        self.en4 = nn.Sequential(
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
        )
        self.en5 = nn.Sequential(
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
        )
        self.en6 = nn.Sequential(
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
        )
        self.en7 = nn.Sequential(
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
        )
        self.en8 = nn.Sequential(
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),
        )

        self.en_att5 = SelfAttention(64)
        self.en_att6 = SelfAttention(128)
        self.en_att7 = SelfAttention(256)
        self.en_att8 = SelfAttention(512)

        # fusion
        self.fs_conv = nn.Conv2d(input_len * 2, 32, kernel_size=4, stride=2, padding=1)
        self.fs2 = nn.Sequential(
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
        )
        self.fs3 = nn.Sequential(
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
        )
        self.fs4 = nn.Sequential(
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
        )
        self.fs5 = nn.Sequential(
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
        )
        self.fs6 = nn.Sequential(
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
        )
        self.fs7 = nn.Sequential(
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
        )

        self.fs_dsrab1 = N2DSRAB(32, expansion=1, num_stacks=1)
        self.fs_dsrab2 = N2DSRAB(32, expansion=1, num_stacks=2)
        self.fs_dsrab3 = N2DSRAB(32, expansion=1, num_stacks=2)
        self.fs_dsrab4 = N2DSRAB(64, expansion=1, num_stacks=4)
        self.fs_dsrab5 = N2DSRAB(64, expansion=1, num_stacks=4)
        self.fs_dsrab6 = N2DSRAB(128, expansion=1, num_stacks=8)
        self.fs_dsrab7 = N2DSRAB(256, expansion=1, num_stacks=8)


        # decoder
        self.dc8 = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(512, out_channels=256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.Dropout2d(inplace=True)
        )
        self.dc7 = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(3*256, out_channels=128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.Dropout2d(inplace=True)
        )
        self.dc6 = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(3*128, out_channels=64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.Dropout2d(inplace=True)
        )
        #
        self.dc5 = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(3*64, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64)
        )
        self.dc5_dsrab = N2DSRAB(in_channels=64, expansion=4, num_stacks=4)
        #
        self.dc4 = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(3*64, out_channels=32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32)
        )
        self.dc4_dsrab = N2DSRAB(in_channels=32, expansion=2, num_stacks=2)
        #
        self.dc3 = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(3*32, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32)
        )
        self.dc3_dsrab = N2DSRAB(in_channels=32, expansion=1, num_stacks=1)
        #
        self.dc2 = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(3*32, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32)
        )
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
        fs1 = self.fs_dsrab1(fs1)
        fs2 = self.fs_dsrab2(fs2)
        fs3 = self.fs_dsrab3(fs3)
        fs4 = self.fs_dsrab4(fs4)
        fs5 = self.fs_dsrab5(fs5)
        fs6 = self.fs_dsrab6(fs6)
        fs7 = self.fs_dsrab7(fs7)

        # Decoder
        dec8 = self.dc8(enc8)
        dec7 = self.dc7(torch.cat([enc7, dec8, fs7], dim=1))
        dec6 = self.dc6(torch.cat([enc6, dec7, fs6], dim=1))
        #
        dec5 = self.dc5(torch.cat([enc5, dec6, fs5], dim=1))
        dec5 = self.dc5_dsrab(dec5) + dec5
        #
        dec4 = self.dc4(torch.cat([enc4, dec5, fs4], dim=1))
        dec4 = self.dc4_dsrab(dec4) + dec4
        #
        dec3 = self.dc3(torch.cat([enc3, dec4, fs3], dim=1))
        dec3 = self.dc3_dsrab(dec3) + dec3
        #
        dec2 = self.dc2(torch.cat([enc2, dec3, fs2], dim=1))
        dec1 = self.dc1(torch.cat([enc1, dec2, fs1], dim=1))

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
    
