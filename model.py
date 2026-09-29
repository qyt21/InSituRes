from typing import Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F




def _sincos_2d_pos_embed(h: int, w: int, dim: int, device, dtype) -> torch.Tensor:
    if dim % 4 != 0:
        raise ValueError(f"pos_embed dim must be divisible by 4, got dim={dim}")
    y = torch.arange(h, device=device, dtype=dtype)
    x = torch.arange(w, device=device, dtype=dtype)
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    yy = yy.reshape(-1, 1)
    xx = xx.reshape(-1, 1)

    omega = torch.arange(dim // 4, device=device, dtype=dtype)
    omega = 1.0 / (10000.0 ** (omega / (dim // 4)))
    out_y = yy * omega.view(1, -1)
    out_x = xx * omega.view(1, -1)

    pe = torch.cat([torch.sin(out_y), torch.cos(out_y), torch.sin(out_x), torch.cos(out_x)], dim=1)
    return pe


def _window_partition(x: torch.Tensor, win: int) -> Tuple[torch.Tensor, Tuple[int, int, int, int]]:
    B, H, W, C = x.shape
    pad_h = (win - (H % win)) % win
    pad_w = (win - (W % win)) % win
    if pad_h != 0 or pad_w != 0:
        x = F.pad(x, (0, 0, 0, pad_w, 0, pad_h), mode="constant", value=0.0)
    Hp, Wp = H + pad_h, W + pad_w
    x = x.view(B, Hp // win, win, Wp // win, win, C)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B * (Hp // win) * (Wp // win), win * win, C)
    return x, (H, W, Hp, Wp)


def _window_reverse(windows: torch.Tensor, win: int, shape_info: Tuple[int, int, int, int], B: int) -> torch.Tensor:
    H, W, Hp, Wp = shape_info
    C = windows.shape[-1]
    x = windows.view(B, Hp // win, Wp // win, win, win, C)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, Hp, Wp, C)
    x = x[:, :H, :W, :].contiguous()
    return x


class WindowTransformerBlock(nn.Module):
    def __init__(self, dim: int, heads: int = 8, mlp_ratio: float = 4.0, drop: float = 0.0, win: int = 8):
        super().__init__()
        self.win = int(win)
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=heads, dropout=drop, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, H, W, C = x.shape

        x1 = self.norm1(x.view(B * H * W, C)).view(B, H, W, C)
        win_tokens, shape_info = _window_partition(x1, self.win)
        a, _ = self.attn(win_tokens, win_tokens, win_tokens, need_weights=False)
        win_tokens = win_tokens + a
        x_attn = _window_reverse(win_tokens, self.win, shape_info, B)

        x = x + x_attn

        x2 = self.norm2(x.view(B * H * W, C)).view(B, H, W, C)
        x = x + self.mlp(x2.view(B * H * W, C)).view(B, H, W, C)
        return x


class SlicePatchTransformer(nn.Module):
    def __init__(self, in_ch: int, dim: int = 192, patch: int = 8, depth: int = 6, heads: int = 8, win: int = 8):
        super().__init__()
        self.patch = int(patch)
        self.win = int(win)
        self.embed = nn.Conv2d(in_ch, dim, kernel_size=self.patch, stride=self.patch, padding=0)
        self.blocks = nn.Sequential(*[WindowTransformerBlock(dim=dim, heads=heads, win=self.win) for _ in range(depth)])
        self.proj = nn.ConvTranspose2d(dim, in_ch, kernel_size=self.patch, stride=self.patch, padding=0)

    def forward(self, x3d: torch.Tensor) -> torch.Tensor:
        B, C, D, H, W = x3d.shape
        p = self.patch

        pad_h = (p - (H % p)) % p
        pad_w = (p - (W % p)) % p
        if pad_h != 0 or pad_w != 0:
            x3d = F.pad(x3d, (0, pad_w, 0, pad_h, 0, 0), mode="reflect")
            H2, W2 = H + pad_h, W + pad_w
        else:
            H2, W2 = H, W

        xs = x3d.permute(0, 2, 1, 3, 4).reshape(B * D, C, H2, W2)
        t = self.embed(xs)
        Hp, Wp = t.shape[-2], t.shape[-1]

        t = t.permute(0, 2, 3, 1).contiguous()
        pe = _sincos_2d_pos_embed(Hp, Wp, t.shape[-1], device=t.device, dtype=t.dtype).view(Hp, Wp, -1)
        t = t + pe.unsqueeze(0)

        t = self.blocks(t)

        t = t.permute(0, 3, 1, 2).contiguous()
        out = self.proj(t)

        out3d = out.reshape(B, D, C, H2, W2).permute(0, 2, 1, 3, 4).contiguous()

        if pad_h != 0 or pad_w != 0:
            out3d = out3d[..., :H, :W].contiguous()

        return out3d


class TransformerSR3D(nn.Module):
    def __init__(self, base: int = 64, up_factor: int = 1, tf_dim: int = 192, tf_depth: int = 6, tf_heads: int = 8, tf_patch: int = 8, tf_win: int = 8):
        super().__init__()
        self.up = 1
        _ = up_factor

        self.head = nn.Conv3d(1, base, 3, padding=1)
        self.trunk = nn.Sequential(
            nn.Conv3d(base, base, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(base, base, 3, padding=1),
        )

        self.tf = SlicePatchTransformer(in_ch=base, dim=tf_dim, patch=tf_patch, depth=tf_depth, heads=tf_heads, win=tf_win)

        self.refine = nn.Sequential(
            nn.Conv3d(base, base, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(base, 1, 3, padding=1),
        )
        nn.init.zeros_(self.refine[-1].weight)
        nn.init.zeros_(self.refine[-1].bias)

        self.upconv = None
        self.ps = None

    def forward(self, x_lr, return_residual: bool = False):
        base = x_lr

        h = self.head(x_lr)
        h = h + self.trunk(h)
        h = self.tf(h)

        res = self.refine(h)
        sr = base + res

        if return_residual:
            return sr, res, base
        return sr


class LearnablePSFKernel2D(nn.Module):
    def __init__(self, ksize: int = 11, init_sigma: float = 1.2):
        super().__init__()
        self.ksize = int(ksize)
        if self.ksize % 2 == 0:
            self.ksize += 1
        rad = self.ksize // 2
        yy, xx = torch.meshgrid(torch.arange(self.ksize), torch.arange(self.ksize), indexing="ij")
        yy = yy.float() - rad
        xx = xx.float() - rad
        g = torch.exp(-(xx ** 2 + yy ** 2) / (2.0 * float(init_sigma) ** 2))
        g = g / (g.sum() + 1e-12)
        self.logits = nn.Parameter(torch.log(g + 1e-12))

    def kernel(self) -> torch.Tensor:
        k = F.softplus(self.logits)
        k = k / (k.sum() + 1e-12)
        return k

    def forward(self, x2d: torch.Tensor) -> torch.Tensor:
        k = self.kernel()
        k = k.view(1, 1, self.ksize, self.ksize)
        rad = self.ksize // 2

        H, W = int(x2d.shape[-2]), int(x2d.shape[-1])
        pad_mode = "reflect" if (H > rad and W > rad) else "replicate"

        x = F.pad(x2d, (rad, rad, rad, rad), mode=pad_mode)
        return F.conv2d(x, k, padding=0)


class ForwardOperatorFastFromLong(nn.Module):
    def __init__(
        self,
        up_factor: int = 1,
        psf_ksize: int = 11,
        init_sigma: float = 1.2,
        min_sigma: float = 1e-6,
    ):
        super().__init__()
        self.up = 1
        _ = up_factor

        self.psf = LearnablePSFKernel2D(ksize=psf_ksize, init_sigma=init_sigma)

        self.gain_raw = nn.Parameter(torch.tensor(0.0, dtype=torch.float32))
        self.bias_raw = nn.Parameter(torch.tensor(0.0, dtype=torch.float32))
        self.bias_max = float(0.15)

        self.log_sigma0 = nn.Parameter(torch.tensor(np.log(0.003), dtype=torch.float32))
        self.log_sigma1 = nn.Parameter(torch.tensor(np.log(0.010), dtype=torch.float32))
        self.min_sigma = float(min_sigma)

    def noise_sigmas(self) -> Tuple[torch.Tensor, torch.Tensor]:
        s0 = torch.exp(self.log_sigma0).clamp_min(self.min_sigma)
        s1 = torch.exp(self.log_sigma1).clamp_min(self.min_sigma)
        return s0, s1

    def gain_eff(self) -> torch.Tensor:
        return torch.exp(0.10 * self.gain_raw)

    def bias_eff(self) -> torch.Tensor:
        return torch.tanh(self.bias_raw) * self.bias_max

    def apply_mu(self, mu_hr: torch.Tensor, lr_hw: Tuple[int, int]) -> torch.Tensor:
        B, C, D, H, W = mu_hr.shape
        x2 = mu_hr.permute(0, 2, 1, 3, 4).reshape(B * D, 1, H, W)

        x2 = self.psf(x2)

        if (x2.shape[-2] != lr_hw[0]) or (x2.shape[-1] != lr_hw[1]):
            x2 = F.interpolate(x2, size=lr_hw, mode="area")

        x2 = x2 * self.gain_eff() + self.bias_eff()

        mu_lr = x2.reshape(B, D, 1, lr_hw[0], lr_hw[1]).permute(0, 2, 1, 3, 4).contiguous()
        return mu_lr

    def nll_gauss(self, mu_lr_pred: torch.Tensor, mu_lr_obs: torch.Tensor) -> torch.Tensor:
        s0, s1 = self.noise_sigmas()
        sigma = (s0 + s1 * mu_lr_pred.detach().abs().sqrt()).clamp_min(self.min_sigma)
        diff = (mu_lr_pred - mu_lr_obs)
        nll = (0.5 * (diff / sigma).pow(2) + torch.log(sigma)).mean()
        return nll

    def beer_consistency(self, mu_lr_pred: torch.Tensor, mu_lr_obs: torch.Tensor) -> torch.Tensor:
        I_pred = torch.exp(-mu_lr_pred.clamp(min=0.0))
        I_obs = torch.exp(-mu_lr_obs.clamp(min=0.0))
        return (I_pred - I_obs).abs().mean()

