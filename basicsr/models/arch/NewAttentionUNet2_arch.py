import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.nn.utils import spectral_norm


class SAM(nn.Module):
    def __init__(self, bias=False):
        super(SAM, self).__init__()
        self.bias = bias
        self.conv = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=7, stride=1, padding=3, dilation=1,
                              bias=self.bias)
        self.layernorm = LayerNorm(1, data_format="channels_first")

    def forward(self, x):
        max = torch.max(x, 1)[0].unsqueeze(1)
        avg = torch.mean(x, 1).unsqueeze(1)
        concat = torch.cat((max, avg), dim=1)
        output = self.conv(concat)
        # layer
        output = self.layernorm(output)
        output = F.sigmoid(output) * x
        return output


class CAM(nn.Module):
    def __init__(self, channels, r=16):
        super(CAM, self).__init__()
        self.channels = channels
        self.r = r
        self.linear = nn.Sequential(
            nn.Linear(in_features=self.channels, out_features=self.channels // self.r, bias=True),
            nn.GELU(),
            nn.Linear(in_features=self.channels // self.r, out_features=self.channels, bias=True))

    def forward(self, x):
        max = F.adaptive_max_pool2d(x, output_size=1)
        avg = F.adaptive_avg_pool2d(x, output_size=1)
        b, c, _, _ = x.size()
        linear_max = self.linear(max.view(b, c)).view(b, c, 1, 1)
        linear_avg = self.linear(avg.view(b, c)).view(b, c, 1, 1)
        output = linear_max + linear_avg
        output = F.sigmoid(output) * x
        return output


class CBAM(nn.Module):
    def __init__(self, channels, r):
        super(CBAM, self).__init__()
        self.channels = channels
        self.r = r
        self.sam = SAM(bias=False)
        self.cam = CAM(channels=self.channels, r=self.r)
        self.conv = nn.Conv2d(channels, channels, 3, 1, 1)
        self.layernorm = LayerNorm(channels, data_format="channels_first")
        self.gelu = nn.GELU()
        self.groupnorm = nn.GroupNorm(num_groups=channels // 8, num_channels=channels)

    def forward(self, x):
        # Conv + layer
        output = self.conv(x)
        # output = self.layernorm(output)
        output = self.groupnorm(output)
        output = self.cam(output)
        # gelu
        output = self.gelu(output)
        output = self.sam(output)
        return output + x


class LayerNorm(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first.
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs
    with shape (batch_size, channels, height, width).
    """

    def __init__(self, normalized_shape, eps=1e-5, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x


class ConvLN(nn.Module):
    def __init__(self, input_channel, hidden_channel, kernel_size=3, stride=1, padding=1, r=2, bias=False,
                 require_act=True):
        super().__init__()
        if require_act:
            self.module = nn.Sequential(
                nn.Conv2d(input_channel, hidden_channel, kernel_size=kernel_size, stride=stride, padding=padding,
                          bias=bias),
                # LayerNorm(hidden_channel, data_format="channels_first"),
                nn.GroupNorm(num_groups=hidden_channel // 8, num_channels=hidden_channel),
                nn.GELU(),
                CBAM(hidden_channel, r=r),
                # nn.MaxPool2d(kernel_size=2, stride=2)
            )
        else:
            self.module = nn.Sequential(
                nn.Conv2d(input_channel, hidden_channel, kernel_size=kernel_size, stride=stride, padding=padding,
                          bias=bias),
                # LayerNorm(hidden_channel, data_format="channels_first"),
                nn.GroupNorm(num_groups=hidden_channel // 8, num_channels=hidden_channel),
                nn.GELU(),
            )

    def forward(self, x):
        # [bs, C, H, W]
        x = self.module(x)
        return x


class SE_Block(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(c, c, kernel_size=1, stride=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return x * self.att(x)


class NewAttentionUNet2(nn.Module):
    """Defines a plain U-Net discriminator with layer normalization (LN)"""

    def __init__(self, input_nc=3, ndf=64):
        super(NewAttentionUNet2, self).__init__()

        self.conv0 = ConvLN(input_nc, ndf, kernel_size=3, stride=1, padding=1, require_act=False)  # 没有注意力
        self.conv1 = ConvLN(ndf, ndf * 2, 4, 2, 1, bias=False)
        self.conv2 = ConvLN(ndf * 2, ndf * 4, 4, 2, 1, bias=False)
        self.conv3 = ConvLN(ndf * 4, ndf * 8, 4, 2, 1, bias=False)
        self.conv4 = ConvLN(ndf * 8, ndf * 8, 4, 2, 1, bias=False)

        # upsample
        self.conv5 = ConvLN(ndf * 8, ndf * 8, 3, 1, 1, bias=False)
        self.conv6 = ConvLN(ndf * 8, ndf * 4, 3, 1, 1, bias=False)
        self.conv7 = ConvLN(ndf * 4, ndf * 2, 3, 1, 1, bias=False)
        self.conv8 = ConvLN(ndf * 2, ndf, 3, 1, 1, bias=False)

        # extra
        self.conv9 = ConvLN(ndf, ndf, 3, 1, 1, bias=False)
        self.conv10 = ConvLN(ndf, ndf, 3, 1, 1, bias=False)

        self.conv11 = nn.Conv2d(ndf, input_nc, 3, 1, 1)
        print('using the Plain UNet')

    def forward(self, x):
        x0 = F.leaky_relu(self.conv0(x), negative_slope=0.2, inplace=False)
        x1 = F.leaky_relu(self.conv1(x0), negative_slope=0.2, inplace=False)
        x2 = F.leaky_relu(self.conv2(x1), negative_slope=0.2, inplace=False)
        x3 = F.leaky_relu(self.conv3(x2), negative_slope=0.2, inplace=False)
        x4 = F.leaky_relu(self.conv4(x3), negative_slope=0.2, inplace=False)
        # upsample
        x4 = F.interpolate(x4, scale_factor=2, mode='bilinear', align_corners=False)
        x5 = F.leaky_relu(self.conv5(x4), negative_slope=0.2, inplace=False)

        x5 = x5 + x3
        x5 = F.interpolate(x5, scale_factor=2, mode='bilinear', align_corners=False)
        x6 = F.leaky_relu(self.conv6(x5), negative_slope=0.2, inplace=False)

        x6 = x6 + x2
        x6 = F.interpolate(x6, scale_factor=2, mode='bilinear', align_corners=False)
        x7 = F.leaky_relu(self.conv7(x6), negative_slope=0.2, inplace=False)

        x7 = x7 + x1
        x7 = F.interpolate(x7, scale_factor=2, mode='bilinear', align_corners=False)
        x8 = F.leaky_relu(self.conv8(x7), negative_slope=0.2, inplace=False)

        x8 = x8 + x0
        # extra
        out = F.leaky_relu(self.conv9(x8), negative_slope=0.2, inplace=False)
        out = F.leaky_relu(self.conv10(out), negative_slope=0.2, inplace=False)
        out = self.conv11(out)

        return out


if __name__ == "__main__":
    from fvcore.nn import FlopCountAnalysis

    model = NewAttentionUNet2().cuda()
    # print(model)
    inputs = torch.randn((1, 3, 256, 256)).cuda()
    flops = FlopCountAnalysis(model, inputs)
    n_param = sum([p.nelement() for p in model.parameters()])  # 所有参数数量

    #  模型的计算复杂度: 48.578 GMac
    # 模型的参数数量: 20.58 M

    # 计算并打印模型的计算复杂度（以GMac为单位）
    gmac = flops.total() / (1024 * 1024 * 1024)
    print(f'模型的计算复杂度: {gmac:.4f} GMac')

    # 计算并打印模型的参数数量（以百万为单位）
    params_million = n_param / (1000 * 1000)
    print(f'模型的参数数量: {params_million:.2f} M')
