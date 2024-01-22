import torch.nn as nn
import torch
import cbam
import math


def batched_index_select(values, indices):
    last_dim = values.shape[-1]
    return values.gather(1, indices[:, :, None].expand(-1, -1, last_dim))


class BasicBlock(nn.Sequential):
    def __init__(
            self, conv, in_channels, out_channels, kernel_size, stride=1, bias=True,
            bn=False, act=nn.PReLU()):

        m = [conv(in_channels, out_channels, kernel_size, bias=bias)]
        if bn:
            m.append(nn.BatchNorm2d(out_channels))
        if act is not None:
            m.append(act)

        super(BasicBlock, self).__init__(*m)


def get_parameters(model, bias):
    for m in model.modules():
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            if bias:
                yield m.bias
            else:
                yield m.weight
        elif isinstance(m, nn.BatchNorm2d):
            if bias:
                yield m.bias
            else:
                yield m.weight


def default_conv(in_channels, out_channels, kernel_size, stride=1, bias=True):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        stride=stride, padding=(kernel_size // 2), bias=bias)


class ResBlock(nn.Module):
    def __init__(
            self, conv, n_feats, kernel_size,
            bias=True, bn=False, act=nn.ReLU(True), res_scale=1):

        super(ResBlock, self).__init__()
        m = []
        for i in range(2):
            m.append(conv(n_feats, n_feats, kernel_size, bias=bias))
            if bn:
                m.append(nn.BatchNorm2d(n_feats))
            if i == 0:
                m.append(act)

        self.body = nn.Sequential(*m)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x).mul(self.res_scale)
        res += x

        return res




class Upsampler(nn.Sequential):
    def __init__(self, scale, n_feats):
        super().__init__()

        m = []
        if (scale & (scale - 1)) == 0:    # Is scale = 2^n?
            for _ in range(int(math.log(scale, 2))):
                m.append(nn.Conv2d(n_feats, 4 * n_feats, 3, 1, 1))
                m.append(nn.PixelShuffle(2))

        elif scale == 3:
            m.append(nn.Conv2d(n_feats, 9 * n_feats, 3, 1, 1))
            m.append(nn.PixelShuffle(3))
        else:
            raise NotImplementedError
        self.body = nn.Sequential(*m)
    def forward(self, x):
        res = self.body(x)
        return res





class invPixelShuffle(nn.Module):

    def __init__(self, ratio=2):
        super(invPixelShuffle, self).__init__()
        self.ratio = ratio

    def forward(self, tensor):
        ratio = self.ratio
        b = tensor.size(0)
        ch = tensor.size(1)
        y = tensor.size(2)
        x = tensor.size(3)
        assert x % ratio == 0 and y % ratio == 0, 'x, y, ratio : {}, {}, {}'.format(x, y, ratio)

        return tensor.view(b, ch, y // ratio, ratio, x // ratio, ratio).permute(0, 1, 3, 5, 2, 4).contiguous().view(b,
                                                                                                                    -1,
                                                                                                                    y // ratio,
                                                                                                                    x // ratio)


class invUpsampler(nn.Sequential):
    def __init__(self, conv, scale, n_feat, bn=False, act=False, bias=True):

        m = []
        if (scale & (scale - 1)) == 0:  # Is scale = 2^n?
            for _ in range(int(math.log(scale, 2))):
                m.append(invPixelShuffle(2))
                m.append(conv(n_feat * 4, n_feat, 3, bias))
                if bn: m.append(nn.BatchNorm2d(n_feat))
                if act: m.append(act())
        elif scale == 3:
            m.append(invPixelShuffle(3))
            m.append(conv(n_feat * 9, n_feat, 3, bias))
            if bn: m.append(nn.BatchNorm2d(n_feat))
            if act: m.append(act())
        else:
            raise NotImplementedError

        super(invUpsampler, self).__init__(*m)

class invUpsampler_module(nn.Module):
    def __init__(self, scale, n_feat, bn=False, act=False, bias=True):
        super(invUpsampler_module, self).__init__()

        self.up = invPixelShuffle(2)
        self.conv = nn.Conv2d(in_channels=n_feat*4, out_channels=n_feat, kernel_size=3,
        stride=1, padding=1) #conv(n_feat*4, n_feat, 3, bias)

    def forward(self, x):
        x = self.up(x)
        # print(x.shape)
        x = self.conv(x)
        return x


class Refine(nn.Module):
    def __init__(self, n_feats, conv=default_conv):
        super(Refine, self).__init__()
        kernel_size = 3
        act = nn.ReLU(True)
        self.conv = nn.Sequential(*[ResBlock(conv, n_feats, kernel_size, act=act),
                                    ResBlock(conv, n_feats, kernel_size, act=act)])

    def forward(self, first, second):
        resdual = second - first
        res = self.conv(resdual)
        res = res + second
        return res

class Multi_scale_fusion_block(nn.Module):
    def __init__(self, n_feats, scale):
        super(Multi_scale_fusion_block, self).__init__()
        self.scale = scale
        if scale == 2:
            self.down1 = nn.Conv2d(in_channels=n_feats, out_channels=n_feats, kernel_size=3, stride=2, padding=1)
            self.down2 = nn.Sequential(
                *[nn.Conv2d(in_channels=n_feats, out_channels=n_feats, kernel_size=3, stride=2, padding=1),
                  nn.ReLU(True),
                  nn.Conv2d(in_channels=n_feats, out_channels=n_feats, kernel_size=3, stride=2, padding=1)])
        elif scale == 4:
            self.down1 = nn.Conv2d(in_channels=n_feats, out_channels=n_feats, kernel_size=3, stride=2, padding=1)
            self.up1 = nn.UpsamplingNearest2d(scale_factor=2)
        elif scale == 8:
            self.up1 = nn.UpsamplingNearest2d(scale_factor=2)
            self.up2 = nn.UpsamplingNearest2d(scale_factor=4)
        self.refine2 = Refine(n_feats)
        self.refine4 = Refine(n_feats)
        self.refine8 = Refine(n_feats)
        #         self.attention = CA(conv=default_conv, n_feats=n_feats, kernel_size=1)
        self.attention = cbam.ChannelGate(n_feats, reduction_ratio=4, pool_types=['avg', 'max', 'var'])
        self.conv = nn.Conv2d(in_channels=n_feats * 3, out_channels=n_feats, kernel_size=1)

    def forward(self, scale2, scale4, scale8, now):
        if self.scale == 2:
            scale4 = self.down1(scale4)
            scale8 = self.down2(scale8)
        elif self.scale == 4:
            scale8 = self.down1(scale8)
            scale2 = self.up1(scale2)
        elif self.scale == 8:
            scale4 = self.up1(scale4)
            scale2 = self.up2(scale2)
        feature1 = self.refine2(scale2, now)
        feature2 = self.refine4(scale4, now)
        feature3 = self.refine8(scale8, now)
        fea = torch.cat((feature1, feature2, feature3), 1)
        fea = self.conv(fea)
        fea = self.attention(fea)
        fea = fea + now
        return fea




class CA(nn.Module):
    def __init__(self, conv, n_feats, kernel_size, bias=True, bn=False, act=nn.ReLU(True), res_scale=1, gama=2, lamb=4,
                 multi=True, spatial=True):

        super(CA, self).__init__()
        m = []
        for i in range(2):
            if i == 0:
                m.append(conv(n_feats, n_feats, kernel_size, bias=bias))
                m.append(act)
            if i == 1:
                if multi:
                    m.append(MUL_small(conv, n_feats))
                else:
                    m.append(conv(n_feats, n_feats, kernel_size, bias=bias))

        # m.append(conv(n_feats, n_feats, kernel_size, bias=bias))
        self.body = nn.Sequential(*m)
        self.attention_layer2 = cbam.ChannelGate(n_feats, reduction_ratio=lamb, pool_types=['avg', 'max', 'var'])

        # self.attention = atten(conv, n_feats)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x)
        res = self.attention_layer2(res)

        res += x

        return res

class MUL_small(nn.Module):
    def __init__(self, conv, n_feats, bias=True):

        super(MUL_small, self).__init__()
        self.n_feats = n_feats
        self.conv1 = conv(n_feats//4, n_feats//4, 1, bias=bias)
        self.conv3 = conv(n_feats//4, n_feats//4, 3, bias=bias)
        self.conv5 = conv(n_feats//4, n_feats//4, 5, bias=bias)
        self.conv7 = conv(n_feats//4, n_feats//4, 7, bias=bias)

    def forward(self, x):
        le = self.n_feats//4
        x1 = x[:, :le, :, :]
        x2 = x[:, le:le*2, :, :]
        x3 = x[:, le*2:le*3, :, :]
        x4 = x[:, le*3:, :, :]
        # print(x1.shape)
        # print(x2.shape)
        # print(x3.shape)
        # print(x4.shape)
        res1 = self.conv1(x1)
        res3 = self.conv3(x2)
        res5 = self.conv5(x3)
        res7 = self.conv7(x4)
        res = torch.cat((res1, res3, res5, res7), 1)

        return res

class PCSR1(nn.Module):
    def __init__(self, conv, n_feats, kernel_size, bias=True, act=nn.ReLU(True), res_scale=1, gama=2, lamb=4):
        super(PCSR1, self).__init__()
        # First branch
        m = []
        for i in range(2):
            if i == 0:
                m.append(conv(n_feats, n_feats, kernel_size, bias=bias))
                m.append(act)
            if i == 1:
                m.append(conv(n_feats, n_feats, kernel_size, bias=bias))

        self.body = nn.Sequential(*m)
        
        self.attention_layer1 = cbam.CSAR_SpatialGate(n_feats, gama=gama)
        
        self.attention_layer2 = cbam.ChannelGate(n_feats, reduction_ratio=lamb, pool_types=['avg', 'max', 'var'])
        self.conv = conv(2 * n_feats, n_feats, 1, bias=bias)
        self.res_scale = res_scale
        # Second branch
        self.conv_feature = nn.Sequential(
            *[nn.Conv2d(in_channels=n_feats, out_channels=n_feats, kernel_size=3, stride=1, padding=1),
              act])
        self.conv_parsing = nn.Sequential(
            *[nn.Conv2d(in_channels=3, out_channels=n_feats, kernel_size=3, stride=1, padding=1),
              act])

        self.conv_fusion = nn.Conv2d(in_channels=n_feats * 2, out_channels=n_feats, kernel_size=3, stride=1,
                                         padding=1)
        self.attention_fusion = nn.Conv2d(in_channels=n_feats * 2, out_channels=n_feats, kernel_size=1, stride=1)

    def forward(self, x, p):
        # First branch
        res = self.body(x)
        res1 = self.attention_layer1(res)
        res2 = self.attention_layer2(res)
        res = torch.cat((res1, res2), 1)
        res = self.conv(res)
        # Second branch
        fea = self.conv_feature(x)
        par = self.conv_parsing(p)
        fea = torch.cat((fea, par), 1)
        fea = self.conv_fusion(fea)
        fea_fusion = torch.cat((fea, res), 1)
        res = self.attention_fusion(fea_fusion)

        res += x

        return res

class Multi_scale_fusion_block4(nn.Module):
    def __init__(self, n_feats, scale):
        super(Multi_scale_fusion_block4, self).__init__()
        self.scale = scale
        if scale ==2:
            self.down1 = nn.Conv2d(in_channels=n_feats, out_channels=n_feats, kernel_size=3, stride=2, padding=1)
        elif scale == 4:
            self.up1 = nn.UpsamplingNearest2d(scale_factor=2)

        self.refine2 = Refine(n_feats)
        self.refine4 = Refine(n_feats)
        self.attention = CA(conv=default_conv, n_feats=n_feats, kernel_size=1)
        self.conv = nn.Conv2d(in_channels=n_feats*2, out_channels=n_feats, kernel_size=1)
    def forward(self, scale2, scale4, now):
        if self.scale ==2:
            scale4 = self.down1(scale4)
        elif self.scale == 4:
            scale2 = self.up1(scale2)
        feature1 = self.refine2(scale2, now)
        feature2 = self.refine4(scale4, now)

        fea = torch.cat((feature1, feature2),1)
        fea = self.conv(fea)
        fea = self.attention(fea)
        fea = fea + now
        return fea
class Multi_scale_fusion_block16(nn.Module):
    def __init__(self, n_feats, scale):
        super(Multi_scale_fusion_block16, self).__init__()
        self.scale = scale
        if scale ==2:
            self.down1 = nn.Conv2d(in_channels=n_feats, out_channels=n_feats, kernel_size=3, stride=2, padding=1)
            self.down2 = nn.Sequential(*[nn.Conv2d(in_channels=n_feats, out_channels=n_feats, kernel_size=3, stride=2, padding=1),
                                         nn.ReLU(True),
                                         nn.Conv2d(in_channels=n_feats, out_channels=n_feats, kernel_size=3, stride=2, padding=1)])
            self.down3 = nn.Sequential(*[nn.Conv2d(in_channels=n_feats, out_channels=n_feats, kernel_size=3, stride=2, padding=1),
                                         nn.ReLU(True),
                                         nn.Conv2d(in_channels=n_feats, out_channels=n_feats, kernel_size=3, stride=2, padding=1),
                                         nn.ReLU(True),
                                         nn.Conv2d(in_channels=n_feats, out_channels=n_feats, kernel_size=3, stride=2,
                                                   padding=1)
                                         ])
        elif scale == 4:
            self.down1 = nn.Conv2d(in_channels=n_feats, out_channels=n_feats, kernel_size=3, stride=2, padding=1)
            self.up1 = nn.UpsamplingNearest2d(scale_factor=2)
            self.down2 = nn.Sequential(
                *[nn.Conv2d(in_channels=n_feats, out_channels=n_feats, kernel_size=3, stride=2, padding=1),
                  nn.ReLU(True),
                  nn.Conv2d(in_channels=n_feats, out_channels=n_feats, kernel_size=3, stride=2, padding=1)])
        elif scale == 8:
            self.up1 = nn.UpsamplingNearest2d(scale_factor=2)
            self.up2 = nn.UpsamplingNearest2d(scale_factor=4)
            self.down1 = nn.Conv2d(in_channels=n_feats, out_channels=n_feats, kernel_size=3, stride=2, padding=1)
        elif scale == 16:
            self.up1 = nn.UpsamplingNearest2d(scale_factor=2)
            self.up2 = nn.UpsamplingNearest2d(scale_factor=4)
            self.up3 = nn.UpsamplingNearest2d(scale_factor=8)
        self.refine2 = Refine(n_feats)
        self.refine4 = Refine(n_feats)
        self.refine8 = Refine(n_feats)
        self.refine16 = Refine(n_feats)
        self.attention = CA(conv=default_conv, n_feats=n_feats, kernel_size=1)
        self.conv = nn.Conv2d(in_channels=n_feats*4, out_channels=n_feats, kernel_size=1)
    def forward(self, scale2, scale4, scale8, scale16, now):
        if self.scale ==2:
            scale4 = self.down1(scale4)
            scale8 = self.down2(scale8)
            scale16 = self.down2(scale16)
        elif self.scale == 4:
            scale8 = self.down1(scale8)
            scale2 = self.up1(scale2)
            scale16 = self.down2(scale16)
        elif self.scale ==8:
            scale4 = self.up1(scale4)
            scale2 = self.up2(scale2)
            scale16 = self.down1(scale16)
        elif self.scale == 16:
            scale4 = self.up1(scale4)
            scale2 = self.up2(scale2)
            scale16 = self.up3(scale16)

        feature1 = self.refine2(scale2, now)
        feature2 = self.refine4(scale4, now)
        feature3 = self.refine8(scale8, now)
        feature4 = self.refine8(scale16, now)
        fea = torch.cat((feature1, feature2, feature3, feature4),1)
        fea = self.conv(fea)
        fea = self.attention(fea)
        fea = fea + now
        return fea
