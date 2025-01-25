import torch
import torch.nn as nn
import torch.nn.functional as F


###############################################################################
#                        TrajGRU Cell Implementation
###############################################################################
class TrajGRUCell(nn.Module):
    """
    A TrajGRU cell that includes a learnable flow mechanism:
    1) Use an extra convolution to estimate optical flow offsets.
    2) Warp the previous hidden state using this flow (here via a simple bilinear sampler).
    3) Compute the gates and candidate hidden state with the warped hidden state.
    """

    def __init__(
        self, input_channels, hidden_channels, kernel_size=3, stride=1, padding=1
    ):
        super(TrajGRUCell, self).__init__()
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels

        # Used to estimate flow from concatenated input + prev hidden
        self.flow_conv = nn.Conv2d(
            in_channels=input_channels + hidden_channels,
            out_channels=2,  # flow field has 2 channels: dx, dy
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        )

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
        # -- store a base grid as a buffer so it is only created once --
        # We initialize to None; we will lazily create it on the first forward
        self.register_buffer("base_grid", None)

    def _get_base_grid(self, B, H, W, device):
        """
        Lazily create the base sampling grid if not already cached or if
        the spatial size has changed (H,W).
        """
        if (
            self.base_grid is None
            or self.base_grid.size(0) != B
            or self.base_grid.size(1) != H
            or self.base_grid.size(2) != W
        ):
            # Create a base grid in [-1,1] x [-1,1]
            base_grid_x = torch.linspace(-1, 1, W, device=device)
            base_grid_y = torch.linspace(-1, 1, H, device=device)
            grid_y, grid_x = torch.meshgrid(base_grid_y, base_grid_x)
            # grid shape: (H, W, 2)
            base_grid = torch.stack((grid_x, grid_y), dim=2)
            # Expand to (B, H, W, 2)
            base_grid = base_grid.unsqueeze(0).expand(B, H, W, 2)
            self.base_grid = base_grid
        return self.base_grid

    def warp(self, hidden, flow):
        """
        Warp hidden state using predicted flow.
        flow: (B, 2, H, W) where flow[:,0] is dx, flow[:,1] is dy.
        This uses a simple grid sampler for bilinear interpolation.
        """
        B, C, H, W = hidden.shape

        # 1) Retrieve or create the cached grid
        base_grid = self._get_base_grid(B, H, W, hidden.device)

        # 2) Convert pixel flow to normalized flow
        flow_x = flow[:, 0] / (W / 2.0)  # (B, H, W)
        flow_y = flow[:, 1] / (H / 2.0)  # (B, H, W)

        # 3) Add flow to base grid
        #    shape of base_grid: (B, H, W, 2)
        #    shape of flow_x,y:  (B, H, W)
        grid_warped = torch.empty_like(base_grid)
        grid_warped[..., 0] = base_grid[..., 0] + flow_x
        grid_warped[..., 1] = base_grid[..., 1] + flow_y

        # 4) Sample
        warped = F.grid_sample(
            hidden,
            grid_warped,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=True,
        )
        return warped

    def forward(self, x, h_prev):
        """
        x:      (B, input_channels,  H, W)
        h_prev: (B, hidden_channels, H, W)
        return: h: updated hidden state
        """
        if h_prev is None:
            h_prev = torch.zeros(
                x.size(0),
                self.hidden_channels,
                x.size(2),
                x.size(3),
                device=x.device,
                dtype=x.dtype,
            )

        # Estimate flow from concatenated x and h_prev
        flow_input = torch.cat([x, h_prev], dim=1)
        flow_field = self.flow_conv(flow_input)  # (B, 2, H, W)

        # Warp previous hidden state
        h_prev_warped = self.warp(h_prev, flow_field)

        # Gating
        combined = torch.cat([x, h_prev_warped], dim=1)  # (B, input+hidden, H, W)
        zr = self.conv_zr(combined)
        z, r = torch.split(zr, self.hidden_channels, dim=1)
        z = torch.sigmoid(z)
        r = torch.sigmoid(r)

        # Candidate hidden
        combined_hr = torch.cat([x, r * h_prev_warped], dim=1)
        h_tilde = torch.tanh(self.conv_h(combined_hr))

        # New hidden
        h = (1 - z) * h_prev_warped + z * h_tilde
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
        B, T, C, H, W = x.shape
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
    def __init__(self, in_channels, out_channels, T):
        super(FirstStageDecoderBlock, self).__init__()
        self.out_channels = out_channels

        self.traj = TrajGRU(in_channels, in_channels)
        self.conv = nn.ConvTranspose2d(
            in_channels * T, out_channels * T, kernel_size=4, stride=2, padding=1
        )
        self.relu = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x, h_init=None):
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
