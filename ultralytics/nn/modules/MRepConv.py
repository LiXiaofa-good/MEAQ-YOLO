import torch
import torch.nn as nn
import torch.nn.functional as F


class MRepConv(nn.Module):
    def __init__(
        self,
        c1: int,
        c2: int,
        stride: int = 2,
        deploy: bool = False,
        use_alpha: bool = True,
    ):
        super().__init__()
        self.c1 = c1
        self.c2 = c2
        self.stride = stride
        self.deploy = deploy
        self.use_alpha = use_alpha

        self.act = nn.SiLU(inplace=True)

        if deploy:
            self.reparam = nn.Conv2d(c1, c2, 3, stride, 1, bias=True)
        else:
            self.branch_3x3 = nn.Sequential(
                nn.Conv2d(c1, c2, kernel_size=3, stride=stride, padding=1, bias=False),
                nn.BatchNorm2d(c2),
            )

            
            self.branch_1x1 = nn.Sequential(
                nn.Conv2d(c1, c2, kernel_size=1, stride=stride, padding=0, bias=False),
                nn.BatchNorm2d(c2),
            )

            
            self.branch_id = (
                nn.BatchNorm2d(c1) if (stride == 1 and c1 == c2) else None
            )
            if use_alpha:
                self.alpha_3x3 = nn.Parameter(torch.tensor(1.0))
                self.alpha_1x1 = nn.Parameter(torch.tensor(1.0))
                self.alpha_id = (
                    nn.Parameter(torch.tensor(1.0))
                    if self.branch_id is not None
                    else None
                )
            else:
                self.register_parameter("alpha_3x3", None)
                self.register_parameter("alpha_1x1", None)
                self.register_parameter("alpha_id", None)

            self._init_weights()

    

    def _init_weights(self):
        n_branches = 2 + (1 if self.branch_id is not None else 0)
        scale = 1.0 / (n_branches ** 0.5)

        for branch in [self.branch_3x3, self.branch_1x1]:
            conv, bn = branch[0], branch[1]
            nn.init.orthogonal_(conv.weight)
            conv.weight.data *= scale
            nn.init.ones_(bn.weight)
            nn.init.zeros_(bn.bias)

        if self.branch_id is not None:
            nn.init.ones_(self.branch_id.weight)
            nn.init.zeros_(self.branch_id.bias)

    

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.deploy:
            return self.act(self.reparam(x))

        if self.use_alpha:
            # ReLU gate keeps alpha positive; +0.1 prevents collapse to zero.
            a3 = F.relu(self.alpha_3x3) + 0.1
            a1 = F.relu(self.alpha_1x1) + 0.1
            out = a3 * self.branch_3x3(x) + a1 * self.branch_1x1(x)
            if self.branch_id is not None:
                aid = F.relu(self.alpha_id) + 0.1
                out = out + aid * self.branch_id(x)
        else:
            out = self.branch_3x3(x) + self.branch_1x1(x)
            if self.branch_id is not None:
                out = out + self.branch_id(x)

        return self.act(out)

   

    @torch.no_grad()
    def switch_to_deploy(self):
        """Fuse all branches into a single Conv2d and switch to deploy mode."""
        if self.deploy:
            return

        kernel, bias = self.get_equivalent_kernel_bias()

        self.reparam = nn.Conv2d(
            in_channels=self.c1,
            out_channels=self.c2,
            kernel_size=3,
            stride=self.stride,
            padding=1,
            bias=True,
        )
        self.reparam.weight.data.copy_(kernel)
        self.reparam.bias.data.copy_(bias)

        # Clean up training-only state
        del self.branch_3x3, self.branch_1x1
        if self.branch_id is not None:
            del self.branch_id
        if self.use_alpha:
            del self.alpha_3x3, self.alpha_1x1
            if self.alpha_id is not None:
                del self.alpha_id

        self.deploy = True

    def get_equivalent_kernel_bias(self):
        k3, b3 = self._fuse_conv_bn(self.branch_3x3)
        k1, b1 = self._fuse_conv_bn(self.branch_1x1)
        k1 = F.pad(k1, [1, 1, 1, 1])  # 1×1 → 3×3

        if self.use_alpha:
            a3 = F.relu(self.alpha_3x3) + 0.1
            a1 = F.relu(self.alpha_1x1) + 0.1
            k3, b3 = k3 * a3, b3 * a3
            k1, b1 = k1 * a1, b1 * a1

        kid, bid = 0, 0
        if self.branch_id is not None:
            kid, bid = self._fuse_identity_bn(self.branch_id)
            if self.use_alpha:
                aid = F.relu(self.alpha_id) + 0.1
                kid, bid = kid * aid, bid * aid

        return k3 + k1 + kid, b3 + b1 + bid


    @staticmethod
    def _fuse_conv_bn(branch: nn.Sequential):
        conv, bn = branch[0], branch[1]
        std = torch.sqrt(bn.running_var + bn.eps)
        t = (bn.weight / std).reshape(-1, 1, 1, 1)
        kernel = conv.weight * t
        bias = bn.bias - bn.running_mean * bn.weight / std
        return kernel, bias

    def _fuse_identity_bn(self, bn: nn.BatchNorm2d):
        c = self.c1
        kernel = torch.zeros((c, c, 3, 3), dtype=bn.weight.dtype, device=bn.weight.device)
        for i in range(c):
            kernel[i, i, 1, 1] = 1.0
        std = torch.sqrt(bn.running_var + bn.eps)
        t = (bn.weight / std)
        kernel = kernel * t.reshape(-1, 1, 1, 1)
        bias = bn.bias - bn.running_mean * bn.weight / std
        return kernel, bias
