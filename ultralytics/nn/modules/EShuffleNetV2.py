import math

import torch
import torch.nn as nn


def channel_shuffle(x, groups):
    batchsize, num_channels, height, width = x.data.size()
    channels_per_group = num_channels // groups
    x = x.view(batchsize, groups, channels_per_group, height, width)
    x = torch.transpose(x, 1, 2).contiguous()
    x = x.view(batchsize, -1, height, width)
    return x


class ECA(nn.Module):
    def __init__(self, channel, b=1, gamma=2):
        super().__init__()
        t = int(abs((math.log(channel, 2) + b) / gamma))
        k = t if t % 2 else t + 1
        k = max(3, k)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k, padding=(k - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.avg_pool(x)  # [B,C,1,1]
        y = y.squeeze(-1).transpose(-1, -2)  # [B,1,C]
        y = self.conv(y).transpose(-1, -2).unsqueeze(-1)  # [B,C,1,1]
        y = self.sigmoid(y)
        return x * y


def build_attn(attn, c):
    if attn is None:
        return nn.Identity()
    if isinstance(attn, str):
        a = attn.lower()
        if a in ("eca",):
            return ECA(c)
        if a in ("none", ""):
            return nn.Identity()
    raise ValueError(f"Unsupported attn={attn}")


def build_act(act):
    if act is None:
        return nn.ReLU(inplace=True)
    if isinstance(act, str):
        a = act.lower()
        if a == "relu":
            return nn.ReLU(inplace=True)
        if a == "silu":
            return nn.SiLU(inplace=True)
        if a in ["hswish", "hardswish"]:
            return nn.Hardswish()
    raise ValueError(f"Unsupported act={act}")


class EShuffleNetV2(nn.Module):
    def __init__(self, inp, oup, stride, attn="eca", act="silu"):
        super().__init__()
        if not (1 <= stride <= 3):
            raise ValueError("illegal stride value")
        self.stride = stride

        act_layer = build_act(act)

        branch_features = oup // 2
        assert (self.stride != 1) or (inp == branch_features << 1)

        if self.stride > 1:
            self.branch1 = nn.Sequential(
                self.depthwise_conv(inp, inp, kernel_size=3, stride=self.stride, padding=1),
                nn.BatchNorm2d(inp),
                nn.Conv2d(inp, branch_features, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(branch_features),
                act_layer,
            )

        self.branch2 = nn.Sequential(
            nn.Conv2d(
                inp if (self.stride > 1) else branch_features,
                branch_features,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=False,
            ),
            nn.BatchNorm2d(branch_features),
            act_layer,
            self.depthwise_conv(branch_features, branch_features, kernel_size=3, stride=self.stride, padding=1),
            nn.BatchNorm2d(branch_features),
            nn.Conv2d(branch_features, branch_features, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(branch_features),
            act_layer,
        )

        self.attn = build_attn(attn, branch_features)

    @staticmethod
    def depthwise_conv(i, o, kernel_size, stride=1, padding=0, bias=False):
        return nn.Conv2d(i, o, kernel_size, stride, padding, bias=bias, groups=i)

    def forward(self, x):
        if self.stride == 1:
            x1, x2 = x.chunk(2, dim=1)
            out2 = self.attn(self.branch2(x2))
            out = torch.cat((x1, out2), dim=1)
        else:
            out1 = self.branch1(x)
            out2 = self.attn(self.branch2(x))
            out = torch.cat((out1, out2), dim=1)

        out = channel_shuffle(out, 2)
        return out
