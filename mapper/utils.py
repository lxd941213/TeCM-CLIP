# coding: UTF-8
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import torch
import os
from collections import OrderedDict
from torch.nn.init import kaiming_normal_


class ApplyNoise(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(channels))
        self.scale = nn.Sequential(
            nn.Conv2d(channels, channels, 1, stride=1, padding=0, bias=False),
            nn.LeakyReLU(0.2),
            nn.Conv2d(channels, channels, 1, stride=1, padding=0, bias=False)
        )
        self.shift = nn.Sequential(
            nn.Conv2d(channels, channels, 1, stride=1, padding=0, bias=False),
            nn.LeakyReLU(0.2),
            nn.Conv2d(channels, channels, 1, stride=1, padding=0, bias=False)
        )
    def forward(self, x, w_t):
        # if noise is None:
        #     noise = torch.randn(x.size(0), 1, x.size(2), x.size(3), device=x.device, dtype=x.dtype)
        w_t = w_t.view(1, 512, 1, 1)
        noise = self.weight.view(1, -1, 1, 1) * torch.randn(x.size(0), 1, x.size(2), x.size(3), device=x.device, dtype=x.dtype).to(x.device)  # 1x512x1x1
        noise = self.scale(w_t) * noise + self.shift(w_t)
        return x + noise


class ApplyStyle(nn.Module):
    """
        @ref: https://github.com/lernapparat/lernapparat/blob/master/style_gan/pytorch_style_gan.ipynb
    """
    def __init__(self, latent_size, channels, use_wscale):
        super(ApplyStyle, self).__init__()
        self.linear = FC(latent_size,
                      channels * 2,
                      gain=1.0,
                      use_wscale=use_wscale)

    def forward(self, x, latent):
        style = self.linear(latent)  # style => [batch_size, n_channels*2]
        shape = [-1, 2, x.size(1), 1, 1]
        style = style.view(shape)    # [batch_size, 2, n_channels, ...]
        x = x * (style[:, 0] + 1.) + style[:, 1]
        return x


class FC(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 gain=2**(0.5),
                 use_wscale=False,
                 lrmul=1.0,
                 bias=True):
        """
            The complete conversion of Dense/FC/Linear Layer of original Tensorflow version.
        """
        super(FC, self).__init__()
        he_std = gain * in_channels ** (-0.5)  # He init
        if use_wscale:
            init_std = 1.0 / lrmul
            self.w_lrmul = he_std * lrmul
        else:
            init_std = he_std / lrmul
            self.w_lrmul = lrmul

        self.weight = torch.nn.Parameter(torch.randn(out_channels, in_channels) * init_std)
        if bias:
            self.bias = torch.nn.Parameter(torch.zeros(out_channels))
            self.b_lrmul = lrmul
        else:
            self.bias = None

    def forward(self, x):
        if self.bias is not None:
            out = F.linear(x, self.weight * self.w_lrmul, self.bias * self.b_lrmul)
        else:
            out = F.linear(x, self.weight * self.w_lrmul)
        out = F.leaky_relu(out, 0.2, inplace=True)
        return out


class Conv2d(nn.Module):
    def __init__(self,
                 input_channels,
                 output_channels,
                 kernel_size,
                 gain=2 ** (0.5),
                 use_wscale=False,
                 lrmul=1,
                 bias=True):
        super().__init__()
        he_std = gain * (input_channels * kernel_size ** 2) ** (-0.5)  # He init
        self.kernel_size = kernel_size
        if use_wscale:
            init_std = 1.0 / lrmul
            self.w_lrmul = he_std * lrmul
        else:
            init_std = he_std / lrmul
            self.w_lrmul = lrmul

        self.weight = torch.nn.Parameter(
            torch.randn(output_channels, input_channels, kernel_size, kernel_size) * init_std)
        if bias:
            self.bias = torch.nn.Parameter(torch.zeros(output_channels))
            self.b_lrmul = lrmul
        else:
            self.bias = None

    def forward(self, x):
        if self.bias is not None:
            return F.conv2d(x, self.weight * self.w_lrmul, self.bias * self.b_lrmul, padding=self.kernel_size // 2)
        else:
            return F.conv2d(x, self.weight * self.w_lrmul, padding=self.kernel_size // 2)


class PixelNorm(nn.Module):
    def __init__(self, epsilon=1e-8):
        """
            @notice: avoid in-place ops.
            https://discuss.pytorch.org/t/encounter-the-runtimeerror-one-of-the-variables-needed-for-gradient-computation-has-been-modified-by-an-inplace-operation/836/3
        """
        super(PixelNorm, self).__init__()
        self.epsilon = epsilon

    def forward(self, x):
        tmp  = torch.mul(x, x) # or x ** 2
        tmp1 = torch.rsqrt(torch.mean(tmp, dim=1, keepdim=True) + self.epsilon)

        return x * tmp1


class InstanceNorm(nn.Module):
    def __init__(self, epsilon=1e-8):
        """
            @notice: avoid in-place ops.
            https://discuss.pytorch.org/t/encounter-the-runtimeerror-one-of-the-variables-needed-for-gradient-computation-has-been-modified-by-an-inplace-operation/836/3
        """
        super(InstanceNorm, self).__init__()
        self.epsilon = epsilon

    def forward(self, x):
        x   = x - torch.mean(x, (2, 3), True)
        tmp = torch.mul(x, x) # or x ** 2
        tmp = torch.rsqrt(torch.mean(tmp, (2, 3), True) + self.epsilon)
        return x * tmp


class LayerEpilogue(nn.Module):
    def __init__(self,
                 channels,
                 dlatent_size,
                 use_wscale,
                 use_noise,
                 use_pixel_norm,
                 use_instance_norm,
                 use_styles):
        super(LayerEpilogue, self).__init__()

        if use_noise:
            self.noise = ApplyNoise(channels)
        self.act = nn.LeakyReLU(negative_slope=0.2)

        if use_pixel_norm:
            self.pixel_norm = PixelNorm()
        else:
            self.pixel_norm = None

        if use_instance_norm:
            self.instance_norm = InstanceNorm()
        else:
            self.instance_norm = None

        if use_styles:
            self.style_mod = ApplyStyle(dlatent_size, channels, use_wscale=use_wscale)
        else:
            self.style_mod = None

    def forward(self, x, w_t):

        x = self.noise(x, w_t)
        x = self.act(x)
        if self.pixel_norm is not None:
            x = self.pixel_norm(x)
        if self.instance_norm is not None:
            x = self.instance_norm(x)
        if self.style_mod is not None:
            x = self.style_mod(x, w_t)

        return x


class AdaIN_FC_Block(nn.Module):
    def __init__(self,
                 use_wscale=True,
                 use_noise=True,
                 use_pixel_norm=False,
                 use_instance_norm=True,
                 noise_input=None,        # noise
                 dlatent_size=512,   # Disentangled latent (W) dimensionality.
                 use_style=True,     # Enable style inputs?
                 ):
        super(AdaIN_FC_Block, self).__init__()

        # noise
        self.noise_input = noise_input

        # A Composition of LayerEpilogue and Conv2d.
        # self.conv_first = Conv2d(dlatent_size, dlatent_size, 1, 1, use_wscale=False)
        self.adaIn1 = LayerEpilogue(dlatent_size, dlatent_size, use_wscale, use_noise,
                                    use_pixel_norm, use_instance_norm, use_style)
        self.conv1 = Conv2d(input_channels=dlatent_size, output_channels=dlatent_size, kernel_size=1, use_wscale=use_wscale)
        self.adaIn2 = LayerEpilogue(dlatent_size, dlatent_size, use_wscale, use_noise,
                                    use_pixel_norm, use_instance_norm, use_style)
        self.conv2 = Conv2d(input_channels=dlatent_size, output_channels=dlatent_size, kernel_size=1, use_wscale=use_wscale)
        self.adaIn3 = LayerEpilogue(dlatent_size, dlatent_size, use_wscale, use_noise,
                                    use_pixel_norm, use_instance_norm, use_style)
        self.conv3 = Conv2d(input_channels=dlatent_size, output_channels=dlatent_size, kernel_size=1, use_wscale=use_wscale)
        self.adaIn4 = LayerEpilogue(dlatent_size, dlatent_size, use_wscale, use_noise,
                                    use_pixel_norm, use_instance_norm, use_style)
        # self.conv_end = Conv2d(dlatent_size, dlatent_size, 3, 1, use_wscale=False)

        self.fc = nn.Sequential(
            FC(dlatent_size, dlatent_size),
            FC(dlatent_size, dlatent_size),
            FC(dlatent_size, dlatent_size),   # 后两层FC看情况决定去掉or不去掉
            FC(dlatent_size, dlatent_size)
        )

    def forward(self, w_i, w_text):
        # shape of w_i b*512, shape of w_text 1*512
        w_text = self.fc(w_text)
        x = w_i.view(w_i.shape[0], w_i.shape[1], 1, 1)
        # x = self.conv_first(x)
        x = self.adaIn1(x, w_text)
        x = self.conv1(x)
        x = self.adaIn2(x, w_text)
        x = self.conv2(x)
        x = self.adaIn3(x, w_text)
        x = self.conv3(x)
        x = self.adaIn4(x, w_text)
        out = x.view(x.shape[0], -1)

        return out


if __name__ == '__main__':
    noise_inputs = []
    for layer_idx in range(18):
        shape = [1, 1, 1, 1]
        noise_inputs.append(torch.randn(*shape).to("cuda"))
    model = AdaIN_FC_Block(noise_input=noise_inputs).cuda()
    w_text = torch.randn(1,512).cuda()
    x = torch.randn(1,512).cuda()
    print(model(x, w_text).shape)

