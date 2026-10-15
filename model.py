import torch
import torch.nn as nn
import torch.nn.functional as F


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = torch.log(torch.tensor(10000.0, device=device)) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        return torch.cat((emb.sin(), emb.cos()), dim=-1)


class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, embed_dim):
        super().__init__()
        self.mlp = nn.Sequential(nn.SiLU(), nn.Linear(embed_dim, out_ch))
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.shortcut = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x, temb):
        h = self.conv1(F.gelu(x))
        h = h + self.mlp(temb)[:, :, None, None]
        h = self.conv2(F.gelu(h))
        return h + self.shortcut(x)


class AttentionBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.group_norm = nn.GroupNorm(32, channels)
        self.q = nn.Conv2d(channels, channels, 1)
        self.k = nn.Conv2d(channels, channels, 1)
        self.v = nn.Conv2d(channels, channels, 1)
        self.proj_out = nn.Conv2d(channels, channels, 1)

    def forward(self, x):
        b, c, h, w = x.shape
        h_ = self.group_norm(x)
        q = self.q(h_).view(b, c, -1)
        k = self.k(h_).view(b, c, -1)
        v = self.v(h_).view(b, c, -1)
        attn = torch.bmm(q.transpose(1, 2), k) * (c ** -0.5)
        attn = F.softmax(attn, dim=-1)
        out = torch.bmm(v, attn.transpose(1, 2)).view(b, c, h, w)
        return x + self.proj_out(out)


class ConvEmbedding(nn.Module):
    def __init__(self, in_channels, out_channels, groups=4):
        super().__init__()
        if out_channels % groups != 0:
            raise ValueError(f"out_channels={out_channels} must be divisible by groups={groups}")
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.GroupNorm(groups, out_channels),
            nn.GELU(),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.GroupNorm(groups, out_channels),
            nn.GELU(),
        )

    def forward(self, x):
        return self.net(x)


class FlowUNet(nn.Module):
    """
    FlowDA velocity network.

    Observation branch:
        innovation = (y_oco - x_ct) * mask
        [innovation, y_oco * mask, mask] -> obs_encoder
    """

    def __init__(
        self,
        state_channels=1,
        obs_channels=3,
        aux_channels=4,
        state_embed_channels=32,
        obs_embed_channels=32,
        aux_embed_channels=16,
        embed_dim=256,
    ):
        super().__init__()

        self.t_emb = SinusoidalPosEmb(embed_dim)
        self.doy_emb = SinusoidalPosEmb(embed_dim)
        self.tod_emb = SinusoidalPosEmb(embed_dim)

        self.time_mlp = nn.Sequential(
            nn.Linear(embed_dim * 3, embed_dim),
            nn.SiLU(),
            nn.Linear(embed_dim, embed_dim),
        )

        self.state_encoder = ConvEmbedding(state_channels, state_embed_channels, groups=4)
        self.obs_encoder = ConvEmbedding(obs_channels, obs_embed_channels, groups=4)
        self.aux_encoder = ConvEmbedding(aux_channels, aux_embed_channels, groups=4)

        init_in_channels = state_embed_channels + obs_embed_channels + aux_embed_channels
        self.init_conv = nn.Conv2d(init_in_channels, 64, 3, padding=1)

        self.down1 = ResBlock(64, 128, embed_dim)
        self.down2 = ResBlock(128, 256, embed_dim)
        self.mid = ResBlock(256, 256, embed_dim)
        self.mid_attn = AttentionBlock(256)
        self.up1 = ResBlock(256 + 256, 128, embed_dim)
        self.up1_attn = AttentionBlock(128)
        self.up2 = ResBlock(128 + 128, 64, embed_dim)
        self.final_conv = nn.Conv2d(64, 1, 3, padding=1)

    def forward(self, xt, x_ct, y_oco, mask, t_fm, doy, tod, aux=None):
        t_feat = self.t_emb(t_fm)
        doy_feat = self.doy_emb(doy)
        tod_feat = self.tod_emb(tod)
        temb = self.time_mlp(torch.cat([t_feat, doy_feat, tod_feat], dim=-1))

        state_feat = self.state_encoder(xt)

        innovation = (y_oco - x_ct) * mask
        obs_input = torch.cat([innovation, y_oco * mask, mask], dim=1)
        obs_feat = self.obs_encoder(obs_input)

        if aux is None:
            raise ValueError("FlowUNet requires auxiliary inputs.")
        if aux.shape[-2:] != xt.shape[-2:]:
            aux = F.interpolate(aux, size=xt.shape[-2:], mode="bilinear", align_corners=False)
        aux_feat = self.aux_encoder(aux)

        x = torch.cat([state_feat, obs_feat, aux_feat], dim=1)
        h1 = self.init_conv(x)

        h2 = self.down1(h1, temb)
        h2_p = F.max_pool2d(h2, 2)

        h3 = self.down2(h2_p, temb)
        h3_p = F.max_pool2d(h3, 2)

        m = self.mid_attn(self.mid(h3_p, temb))

        u1 = F.interpolate(m, size=h3.shape[-2:], mode="bilinear", align_corners=False)
        u1 = self.up1_attn(self.up1(torch.cat([u1, h3], dim=1), temb))

        u2 = F.interpolate(u1, size=h2.shape[-2:], mode="bilinear", align_corners=False)
        u2 = self.up2(torch.cat([u2, h2], dim=1), temb)

        return self.final_conv(u2)
