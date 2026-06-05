from __future__ import annotations

import math
from typing import Literal

import torch
import torch.nn as nn

from ultralytics.utils.tal import dist2bbox, make_anchors

__all__ = ["QDDHead"]

RegBlockType = Literal["conv3", "fullwidth"]
ClsDWType = Literal["dw3", "dw5", "parallel"]
AttType = Literal["none", "ca"]


def autopad(k, p=None, d: int = 1):
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]
    return p


def _gn_groups(c: int) -> int:
    for g in (8, 4, 2, 1):
        if c % g == 0:
            return g
    return 1


class Conv(nn.Module):
    """Conv2d → BN → SiLU。."""

    default_act = nn.SiLU()

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True, bn=True):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=not bn)
        self.bn = nn.BatchNorm2d(c2) if bn else nn.Identity()
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        return self.act(self.conv(x))


class DFL(nn.Module):
    """Distribution Focal Loss 积分解码。."""

    def __init__(self, c1=16):
        super().__init__()
        self.conv = nn.Conv2d(c1, 1, 1, bias=False).requires_grad_(False)
        self.conv.weight.data[:] = torch.arange(c1, dtype=torch.float).view(1, c1, 1, 1)
        self.c1 = c1

    def forward(self, x):
        b, _c, a = x.shape
        return self.conv(x.view(b, 4, self.c1, a).transpose(2, 1).softmax(1)).view(b, 4, a)


class CoordAtt(nn.Module):
    """CoordAttention H+W."""

    def __init__(self, channels, reduction=16):
        super().__init__()
        mid = max(8, channels // reduction)
        ng = _gn_groups(mid)
        self.conv1 = nn.Conv2d(channels, mid, 1, bias=False)
        self.gn = nn.GroupNorm(ng, mid)
        self.act = nn.SiLU(inplace=True)
        self.conv_h = nn.Conv2d(mid, channels, 1, bias=True)
        self.conv_w = nn.Conv2d(mid, channels, 1, bias=True)

    def forward(self, x):
        _b, _c, h, _w = x.shape
        x_h = x.mean(dim=3, keepdim=True)
        x_w = x.mean(dim=2, keepdim=True).transpose(2, 3)
        y = self.act(self.gn(self.conv1(torch.cat([x_h, x_w], dim=2))))
        a_h = torch.sigmoid(self.conv_h(y[:, :, :h]))
        a_w = torch.sigmoid(self.conv_w(y[:, :, h:].transpose(2, 3)))
        return x * a_h * a_w


def build_attention(att_type: AttType, channels: int, reduction: int = 16) -> nn.Module:
    m = {
        "none": nn.Identity,
        "ca": lambda: CoordAtt(channels, reduction),
    }
    if att_type not in m:
        raise ValueError(f"未知 att_type='{att_type}'，可选: {list(m)}")
    factory = m[att_type]
    return factory() if callable(factory) and factory is not nn.Identity else nn.Identity()


class RegBlockConv3(nn.Module):
    def __init__(self, c: int):
        super().__init__()
        self.dw = Conv(c, c, k=3, g=c)  # DW-3×3 + BN + SiLU
        self.pw = Conv(c, c, k=1)  # PW    + BN + SiLU

    def forward(self, x):
        return self.pw(self.dw(x))


class RegBlockFullWidth(nn.Module):
    def __init__(self, c: int):
        super().__init__()
        self.dw_d1 = Conv(c, c, k=3, g=c, d=1, act=False)  # BN only
        self.dw_d2 = Conv(c, c, k=3, g=c, d=2, act=False)  # BN only
        self.bn_fuse = nn.BatchNorm2d(c)
        self.act = nn.SiLU(inplace=True)
        self.pw = Conv(c, c, k=1, act=False)  # BN only
        self.out_act = nn.SiLU(inplace=True)

    def forward(self, x):
        y = self.act(self.bn_fuse(self.dw_d1(x) + self.dw_d2(x)))
        return self.out_act(self.pw(y) + x)


def build_reg_block(reg_block_type: RegBlockType, c: int) -> nn.Module:
    if reg_block_type == "conv3":
        return RegBlockConv3(c)
    if reg_block_type == "fullwidth":
        return RegBlockFullWidth(c)
    raise ValueError(f"未知 reg_block_type='{reg_block_type}'，可选: conv3 / fullwidth")


class ClsBlock(nn.Module):
    def __init__(
        self,
        c_in: int,
        c_out: int,
        cls_dw_type: ClsDWType = "parallel",
        att_type: AttType = "ca",
        reduction: int = 16,
    ):
        super().__init__()
        self.cls_dw_type = cls_dw_type

        if cls_dw_type == "dw3":
            self.dw = Conv(c_in, c_in, k=3, g=c_in, act=False)
            self.bn_fuse = nn.Identity()
            self.act = nn.SiLU(inplace=True)

        elif cls_dw_type == "dw5":
            self.dw = Conv(c_in, c_in, k=5, g=c_in, act=False)
            self.bn_fuse = nn.Identity()
            self.act = nn.SiLU(inplace=True)

        elif cls_dw_type == "parallel":
            self.dw_d1 = Conv(c_in, c_in, k=3, g=c_in, d=1, act=False)
            self.dw_d2 = Conv(c_in, c_in, k=3, g=c_in, d=2, act=False)
            self.bn_fuse = nn.BatchNorm2d(c_in)
            self.act = nn.SiLU(inplace=True)

        else:
            raise ValueError(f"未知 cls_dw_type='{cls_dw_type}'，dw3 / dw5 / parallel")

        self.pw = Conv(c_in, c_out, k=1)  # BN + SiLU
        self.att = build_attention(att_type, c_out, reduction)

    def forward(self, x):
        if self.cls_dw_type in ("dw3", "dw5"):
            y = self.act(self.bn_fuse(self.dw(x)))
        else:  # parallel
            y = self.act(self.bn_fuse(self.dw_d1(x) + self.dw_d2(x)))
        return self.att(self.pw(y))


class QDDHead(nn.Module):
    dynamic: bool = False
    export: bool = False
    shape: tuple | None = None
    anchors: torch.Tensor = torch.empty(0)
    strides: torch.Tensor = torch.empty(0)

    def __init__(
        self,
        nc: int = 80,
        reg_max: int = 16,
        hidc: int = 64,
        clsc: int = 64,
        ch: tuple = (),
        reg_block_type: RegBlockType = "fullwidth",
        cls_dw_type: ClsDWType = "parallel",
        att_type: AttType = "ca",
        use_quality_gate: bool = True,
        reduction: int = 16,
    ):
        super().__init__()
        assert len(ch) > 0, "ch must not be empty"
        self.nc = nc
        self.nl = len(ch)
        self.reg_max = reg_max
        self.no = nc + reg_max * 4
        self.stride = torch.zeros(self.nl)
        self.use_quality_gate = use_quality_gate
        self.reg_block_type = reg_block_type
        self.cls_dw_type = cls_dw_type
        self.att_type = att_type

        _min_ch = min(ch)
        hidc = min(hidc, _min_ch)
        clsc = min(clsc, _min_ch)
        self.hidc = hidc
        self.clsc = clsc

        self.reg_stems = nn.ModuleList([Conv(c, hidc, k=1) for c in ch])

        self.reg_blocks = nn.ModuleList([build_reg_block(reg_block_type, hidc) for _ in ch])

        self.quality_proj = nn.ModuleList([nn.Conv2d(hidc, 1, 1, bias=True) for _ in ch]) if use_quality_gate else None

        self.reg_out = nn.ModuleList([nn.Conv2d(hidc, 4 * reg_max, 1, bias=True) for _ in ch])

        self.cls_blocks = nn.ModuleList(
            [ClsBlock(c_in=c, c_out=clsc, cls_dw_type=cls_dw_type, att_type=att_type, reduction=reduction) for c in ch]
        )

        self.cls_out = nn.ModuleList([nn.Conv2d(clsc, nc, 1, bias=True) for _ in ch])

        self.dfl = DFL(self.reg_max) if self.reg_max > 1 else nn.Identity()

    def forward(self, x: list[torch.Tensor]):
        shape = x[0].shape

        for i in range(self.nl):
            reg_feat = self.reg_blocks[i](self.reg_stems[i](x[i]))
            reg = self.reg_out[i](reg_feat)

            cls_feat = self.cls_blocks[i](x[i])

            if self.use_quality_gate:
                quality_logit = self.quality_proj[i](reg_feat)  # (B,1,H,W)
                cls_feat = cls_feat * torch.sigmoid(quality_logit)

            cls = self.cls_out[i](cls_feat)
            x[i] = torch.cat((reg, cls), dim=1)

        if self.training:
            return x

        if self.dynamic or self.shape != shape:
            self.anchors, self.strides = (t.transpose(0, 1) for t in make_anchors(x, self.stride, 0.5))
            self.shape = shape

        x_cat = torch.cat([xi.view(shape[0], self.no, -1) for xi in x], dim=2)

        if self.export and getattr(self, "format", None) in ("saved_model", "pb", "tflite", "edgetpu", "tfjs"):
            box = x_cat[:, : self.reg_max * 4]
            cls = x_cat[:, self.reg_max * 4 :]
        else:
            box, cls = x_cat.split((self.reg_max * 4, self.nc), dim=1)

        dbox = dist2bbox(self.dfl(box), self.anchors.unsqueeze(0), xywh=True, dim=1) * self.strides

        if self.export and getattr(self, "format", None) in ("tflite", "edgetpu"):
            img_h = shape[2] * self.stride[0]
            img_w = shape[3] * self.stride[0]
            img_size = torch.tensor([img_w, img_h, img_w, img_h], device=dbox.device).reshape(1, 4, 1)
            dbox = dbox / img_size

        y = torch.cat((dbox, cls.sigmoid()), dim=1)
        return y if self.export else (y, x)

    def bias_init(self) -> None:
        for i, (reg_o, cls_o, s) in enumerate(zip(self.reg_out, self.cls_out, self.stride)):
            reg_o.bias.data.fill_(1.0)
            if self.use_quality_gate:
                self.quality_proj[i].bias.data.fill_(4.0)
            s_val = float(s)
            cls_o.bias.data[:] = math.log(5.0 / self.nc / (640.0 / s_val) ** 2) if s_val > 0 else 0.0

    def fuse(self) -> QDDHead:
        for m in self.modules():
            if isinstance(m, Conv) and not isinstance(m.bn, nn.Identity):
                m.conv = _fuse_conv_bn(m.conv, m.bn)
                m.bn = nn.Identity()
                m.forward = m.forward_fuse  # type: ignore
        return self

    def extra_repr(self) -> str:
        return (
            f"nc={self.nc}, nl={self.nl}, reg_max={self.reg_max}, "
            f"hidc={self.hidc}, clsc={self.clsc}, "
            f"reg_block='{self.reg_block_type}', "
            f"cls_dw='{self.cls_dw_type}', att='{self.att_type}', "
            f"quality_gate={self.use_quality_gate}"
        )


def _fuse_conv_bn(conv: nn.Conv2d, bn: nn.BatchNorm2d) -> nn.Conv2d:
    std = (bn.running_var + bn.eps).sqrt()
    scale = (bn.weight / std).reshape(-1, 1, 1, 1)
    fused = nn.Conv2d(
        conv.in_channels,
        conv.out_channels,
        conv.kernel_size,
        conv.stride,
        conv.padding,
        groups=conv.groups,
        dilation=conv.dilation,
        bias=True,
    )
    fused.weight.data = conv.weight.data * scale
    fused.bias.data = bn.bias.data - bn.running_mean.data * bn.weight.data / std
    return fused
