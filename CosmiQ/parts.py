# parts_bn.py
import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# Deterministic blocks (your original code)
# ============================================================

class double_conv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(type(self), self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            #nn.BatchNorm2d(out_ch, momentum=0.005),
            nn.GroupNorm(num_groups=8, num_channels=out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            #nn.BatchNorm2d(out_ch, momentum=0.005),
            nn.GroupNorm(num_groups=8, num_channels=out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class inconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(type(self), self).__init__()
        self.conv = double_conv(in_ch, out_ch)

    def forward(self, x):
        x = self.conv(x)
        return x


class down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(type(self), self).__init__()
        self.mpconv = nn.Sequential(
            nn.MaxPool2d(2),
            double_conv(in_ch, out_ch)
        )

    def forward(self, x):
        x = self.mpconv(x)
        return x


class up(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(type(self), self).__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, 2, stride=2)
        self.conv = double_conv(in_ch, out_ch)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x


class outconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(type(self), self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 1)

    def forward(self, x):
        x = self.conv(x)
        return x


# ============================================================
# Variational / Bayesian building blocks (for VI U-Net)
# ============================================================

class BayesianConv2d(nn.Module):
    """
    Conv2d with mean-field Gaussian posterior over weights:
        w ~ N(mu, sigma^2), sigma = softplus(rho).

    Provides kl_divergence() for ELBO training.
    """
    def __init__(self,
                 in_ch,
                 out_ch,
                 kernel_size=3,
                 stride=1,
                 padding=1,
                 dilation=1,
                 groups=1,
                 bias=True,
                 sigma_prior=1.0):
        super(BayesianConv2d, self).__init__()

        self.in_ch = in_ch
        self.out_ch = out_ch
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.bias_flag = bias
        self.sigma_prior = float(sigma_prior)

        # variational parameters for weights
        self.weight_mu = nn.Parameter(
            torch.Tensor(out_ch, in_ch // groups, *self.kernel_size)
        )
        self.weight_rho = nn.Parameter(
            torch.Tensor(out_ch, in_ch // groups, *self.kernel_size)
        )

        if self.bias_flag:
            self.bias_mu = nn.Parameter(torch.Tensor(out_ch))
            self.bias_rho = nn.Parameter(torch.Tensor(out_ch))
        else:
            self.register_parameter("bias_mu", None)
            self.register_parameter("bias_rho", None)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_normal_(self.weight_mu, mode="fan_out", nonlinearity="relu")
        nn.init.constant_(self.weight_rho, -5.0)  # small initial sigma
        if self.bias_flag:
            nn.init.constant_(self.bias_mu, 0.0)
            nn.init.constant_(self.bias_rho, -5.0)

    def _sample_weights(self):
        # sigma_q = softplus(rho)
        sigma_w = F.softplus(self.weight_rho)
        eps_w = torch.randn_like(self.weight_mu)
        weight = self.weight_mu + sigma_w * eps_w

        if self.bias_flag:
            sigma_b = F.softplus(self.bias_rho)
            eps_b = torch.randn_like(self.bias_mu)
            bias = self.bias_mu + sigma_b * eps_b
        else:
            bias = None

        return weight, bias

    def kl_divergence(self):
        """
        KL( N(mu_q, sigma_q^2) || N(0, sigma_p^2) ), summed over all params.
        """
        sigma_q = F.softplus(self.weight_rho)
        sigma_p = self.sigma_prior

        # weights
        kl_weight = (
            torch.log(sigma_p / sigma_q)
            + (sigma_q**2 + self.weight_mu**2) / (2 * sigma_p**2)
            - 0.5
        ).sum()

        # bias
        if self.bias_flag:
            sigma_q_b = F.softplus(self.bias_rho)
            kl_bias = (
                torch.log(sigma_p / sigma_q_b)
                + (sigma_q_b**2 + self.bias_mu**2) / (2 * sigma_p**2)
                - 0.5
            ).sum()
        else:
            kl_bias = 0.0

        return kl_weight + kl_bias

    def forward(self, x):
        weight, bias = self._sample_weights()
        return F.conv2d(
            x,
            weight,
            bias,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups
        )


def _make_groupnorm(out_ch, max_groups=8):
    """
    Helper to choose a valid number of groups for GroupNorm.
    Ensures out_ch is divisible by num_groups.
    """
    num_groups = min(max_groups, out_ch)
    while out_ch % num_groups != 0 and num_groups > 1:
        num_groups -= 1
    return nn.GroupNorm(num_groups=num_groups, num_channels=out_ch)


class double_conv_vi(nn.Module):
    """
    Variational double conv: (BayesianConv2d + GN + ReLU) x 2
    """
    def __init__(self, in_ch, out_ch, sigma_prior=1.0):
        super(double_conv_vi, self).__init__()
        self.conv1 = BayesianConv2d(in_ch, out_ch, kernel_size=3, padding=1,
                                    sigma_prior=sigma_prior)
        self.gn1   = _make_groupnorm(out_ch)
        self.conv2 = BayesianConv2d(out_ch, out_ch, kernel_size=3, padding=1,
                                    sigma_prior=sigma_prior)
        self.gn2   = _make_groupnorm(out_ch)
        self.relu  = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.relu(self.gn1(self.conv1(x)))
        x = self.relu(self.gn2(self.conv2(x)))
        return x


class inconv_vi(nn.Module):
    def __init__(self, in_ch, out_ch, sigma_prior=1.0):
        super(inconv_vi, self).__init__()
        self.conv = double_conv_vi(in_ch, out_ch, sigma_prior=sigma_prior)

    def forward(self, x):
        return self.conv(x)


class down_vi(nn.Module):
    def __init__(self, in_ch, out_ch, sigma_prior=1.0):
        super(down_vi, self).__init__()
        self.mp   = nn.MaxPool2d(2)
        self.conv = double_conv_vi(in_ch, out_ch, sigma_prior=sigma_prior)

    def forward(self, x):
        x = self.mp(x)
        x = self.conv(x)
        return x


class up_vi(nn.Module):
    """
    Variational up block, mirroring your deterministic `up` block:

    - in_ch: channels AFTER concatenation (skip + upsampled)
    - out_ch: output channels of this block

    For hidden=32 in a 2-level U-Net:
      x1 (deep) has 2*hidden channels
      x2 (skip) has hidden channels
      so you instantiate: up_vi(in_ch=2*hidden, out_ch=hidden)
    """
    def __init__(self, in_ch, out_ch, sigma_prior=1.0):
        super(up_vi, self).__init__()
        # x1 has in_ch channels before upsampling here
        self.up = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)
        # after upsampling: channels = out_ch; concat with skip (out_ch) -> in_ch
        self.conv = double_conv_vi(in_ch, out_ch, sigma_prior=sigma_prior)

    def forward(self, x1, x2):
        # x1: deeper feature (e.g. [B, 2*hidden, H/2, W/2])
        # x2: skip connection (e.g. [B, hidden, H,   W])
        x1 = self.up(x1)                 # [B, out_ch, H,   W]
        x  = torch.cat([x2, x1], dim=1)  # [B, out_ch+out_ch = in_ch, H, W]
        x  = self.conv(x)
        return x
    
