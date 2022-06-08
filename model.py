import model.common as common
import torch.nn.functional as F
import torch.nn as nn
import torch


def fish_block(args, conv=common.default_conv, n_feats=64, multi_scale=False, multi_small=False, CSCR=False,
               CSCR1=False, spatial=True, CA=False, SA=False, PCSR=False, PCSR1=False):
    kernel_size = 3
    res = []
    act = nn.ReLU(True)
    # if multi_scale:
    #     body.append(common.MUL(conv, n_feats))
    # elif multi_small:
    #     body.append(common.MUL_small(conv, n_feats))

    if CSCR:
        res.append(common.CSCR1(
            conv, n_feats, kernel_size, act=act, res_scale=args.res_scale, multi=multi_scale, spatial=spatial
        ))
        res.append(common.CSCR1(
            conv, n_feats, kernel_size, act=act, res_scale=args.res_scale, multi=multi_scale, spatial=spatial
        ))
    elif CSCR1:
        res.append(common.CSCR(
            conv, n_feats, kernel_size, act=act, res_scale=args.res_scale, multi=multi_scale, spatial=spatial
        ))
        res.append(common.CSCR(
            conv, n_feats, kernel_size, act=act, res_scale=args.res_scale, multi=multi_scale, spatial=spatial
        ))
    elif CA:
        res.append(common.CA(
            conv, n_feats, kernel_size, act=act, res_scale=args.res_scale, multi=multi_scale, spatial=spatial
        ))
        res.append(common.CA(
            conv, n_feats, kernel_size, act=act, res_scale=args.res_scale, multi=multi_scale, spatial=spatial
        ))
    elif SA:
        res.append(common.SA(
            conv, n_feats, kernel_size, act=act, res_scale=args.res_scale, multi=multi_scale, spatial=spatial
        ))
        res.append(common.SA(
            conv, n_feats, kernel_size, act=act, res_scale=args.res_scale, multi=multi_scale, spatial=spatial
        ))
    elif PCSR:
        res.append(common.PCSR(
            conv, n_feats, kernel_size, act=act, res_scale=args.res_scale, multi=multi_scale, spatial=spatial
        ))
        res.append(common.PCSR(
            conv, n_feats, kernel_size, act=act, res_scale=args.res_scale, multi=multi_scale, spatial=spatial
        ))
    elif PCSR1:
        res.append(common.PCSR1(
            conv, n_feats, kernel_size, act=act, res_scale=args.res_scale, multi=multi_scale, spatial=spatial
        ))
        res.append(common.PCSR1(
            conv, n_feats, kernel_size, act=act, res_scale=args.res_scale, multi=multi_scale, spatial=spatial
        ))
    else:
        res.append(common.ResBlock(conv, n_feats, kernel_size, act=act, res_scale=args.res_scale))
        res.append(common.ResBlock(conv, n_feats, kernel_size, act=act, res_scale=args.res_scale))
    return res


class ldfishnet(nn.Module):
    def __init__(self, args, pretrained_ld=False):
        super(ldfishnet, self).__init__()
        self.parsing = LD(args)
        if pretrained_ld:
            self.parsing.load_state_dict(torch.load('./model/parsingNet.pth'))
        self.sr = FISHNET(args)

    def forward(self, x):
        p = self.parsing(x)
        sr = self.sr(x, p)
        return sr


class RefineModule(nn.Module):
    def __init__(self, n_feats, conv=common.default_conv):
        super(RefineModule, self).__init__()
        kernel_size = 3
        act = nn.ReLU(True)
        self.conv = nn.Sequential(*[common.ResBlock(conv, n_feats, kernel_size, act=act),
                                    common.ResBlock(conv, n_feats, kernel_size, act=act)])

    def forward(self, first, second):
        resdual = second - first
        res = self.conv(resdual)
        res = res + second
        return res


class RefineModule1(nn.Module):
    def __init__(self, n_feats, conv=common.default_conv):
        super(RefineModule1, self).__init__()
        kernel_size = 3
        act = nn.ReLU(True)
        self.conv = nn.Sequential(*[common.ResBlock(conv, n_feats, kernel_size, act=act),
                                    common.ResBlock(conv, n_feats, kernel_size, act=act)])

    def forward(self, first, second):
        resdual = first
        res = self.conv(resdual)
        res = res + second
        return res


class FISHNET(nn.Module):
    def __init__(self, args, conv=common.default_conv):
        super(FISHNET, self).__init__()

        n_resblocks = 8
        n_feats = 64
        kernel_size = 3
        scale = 8
        act = nn.ReLU(True)
        self.args = args
        self.sub_mean = common.MeanShift(args.rgb_range)
        self.add_mean = common.MeanShift(args.rgb_range, sign=1)

        # define head module
        m_head = [conv(args.n_colors, n_feats, kernel_size)]
        if args.refine:
            self.refine = nn.Sequential(*[RefineModule1(n_feats),
                                          RefineModule1(n_feats),
                                          RefineModule1(n_feats),
                                          RefineModule1(n_feats),
                                          RefineModule1(n_feats),
                                          RefineModule1(n_feats)
                                          ])
        elif args.refine1:
            self.refine1 = nn.Sequential(*[common.Bottleneck(inplanes=n_feats * 2, planes=n_feats),
                                           common.Bottleneck(inplanes=n_feats * 2, planes=n_feats),
                                           common.Bottleneck(inplanes=n_feats * 2, planes=n_feats),
                                           common.Bottleneck(inplanes=n_feats * 2, planes=n_feats),
                                           common.Bottleneck(inplanes=n_feats * 2, planes=n_feats),
                                           common.Bottleneck(inplanes=n_feats * 2, planes=n_feats)])
        elif args.refine2:
            self.refine2 = nn.Sequential(*[common.Multi_scale_fusion_block(n_feats, scale=8),
                                          common.Multi_scale_fusion_block(n_feats, scale=4),
                                          common.Multi_scale_fusion_block(n_feats, scale=2),
                                          common.Multi_scale_fusion_block(n_feats, scale=2),
                                          common.Multi_scale_fusion_block(n_feats, scale=4),
                                          common.Multi_scale_fusion_block(n_feats, scale=8),
                                          ])
        # define body module

        self.up1 = nn.Sequential(*common.Upsampler(conv, 2, n_feats, act=False))
        if self.args.concat_head1:
            self.concat_conv1 = nn.Conv2d(in_channels=n_feats + 3, out_channels=n_feats, kernel_size=1, stride=1)
        self.up_stage1 = nn.Sequential(
            *fish_block(args, n_feats=args.n_feats, multi_scale=args.multi, CSCR=args.CSCR, CSCR1=args.CSCR1,
                        spatial=args.spatial, CA=args.CA, SA=args.SA, PCSR=args.PCSR, PCSR1=args.PCSR1))
        self.up2 = nn.Sequential(*common.Upsampler(conv, 2, n_feats, act=False))
        if self.args.concat_head2:
            self.concat_conv2 = nn.Conv2d(in_channels=n_feats + 3, out_channels=n_feats, kernel_size=1, stride=1)
        self.up_stage2 = nn.Sequential(
            *fish_block(args, n_feats=args.n_feats, multi_scale=args.multi, CSCR=args.CSCR, CSCR1=args.CSCR1,
                        spatial=args.spatial, CA=args.CA, SA=args.SA, PCSR=args.PCSR, PCSR1=args.PCSR1))
        self.up3 = nn.Sequential(*common.Upsampler(conv, 2, n_feats, act=False))
        if self.args.concat_head3:
            self.concat_conv3 = nn.Conv2d(in_channels=n_feats + 3, out_channels=n_feats, kernel_size=1, stride=1)
        self.up_stage3 = nn.Sequential(
            *fish_block(args, n_feats=args.n_feats, multi_scale=args.multi, CSCR=args.CSCR, CSCR1=args.CSCR1,
                        spatial=args.spatial, CA=args.CA, SA=args.SA, PCSR=args.PCSR, PCSR1=args.PCSR1))

        self.down1 = nn.Sequential(*common.invUpsampler(conv, 2, n_feats, act=False))
        if self.args.concat_body1:
            self.concat_conv3 = nn.Conv2d(in_channels=n_feats + 3, out_channels=n_feats, kernel_size=1, stride=1)
        self.down_stage1 = nn.Sequential(
            *fish_block(args, n_feats=args.n_feats, multi_scale=args.multi, CSCR=args.CSCR, CSCR1=args.CSCR1,
                        spatial=args.spatial, CA=args.CA, SA=args.SA, PCSR=args.PCSR, PCSR1=args.PCSR1))
        self.down2 = nn.Sequential(*common.invUpsampler(conv, 2, n_feats, act=False))
        if self.args.concat_body2:
            self.concat_conv2 = nn.Conv2d(in_channels=n_feats + 3, out_channels=n_feats, kernel_size=1, stride=1)
        self.down_stage2 = nn.Sequential(
            *fish_block(args, n_feats=args.n_feats, multi_scale=args.multi, CSCR=args.CSCR, CSCR1=args.CSCR1,
                        spatial=args.spatial, CA=args.CA, SA=args.SA, PCSR=args.PCSR, PCSR1=args.PCSR1))
        self.down3 = nn.Sequential(*common.invUpsampler(conv, 2, n_feats, act=False))
        if self.args.concat_body3:
            self.concat_conv1 = nn.Conv2d(in_channels=n_feats + 3, out_channels=n_feats, kernel_size=1, stride=1)
        self.down_stage3 = nn.Sequential(
            *fish_block(args, n_feats=args.n_feats, multi_scale=args.multi, CSCR=args.CSCR, CSCR1=args.CSCR1,
                        spatial=args.spatial, CA=args.CA, SA=args.SA, PCSR=args.PCSR, PCSR1=args.PCSR1))

        self.conv_tail1 = nn.Conv2d(in_channels=n_feats * 2, out_channels=n_feats, kernel_size=1, stride=1)
        self.conv = conv(n_feats, n_feats, 3)
        self.up21 = nn.Sequential(*common.Upsampler(conv, 2, n_feats, act=False))
        if self.args.concat_tail1:
            self.concat_conv1 = nn.Conv2d(in_channels=n_feats + 3, out_channels=n_feats, kernel_size=1, stride=1)
        self.conv_tail2 = nn.Conv2d(in_channels=n_feats * 2, out_channels=n_feats, kernel_size=1, stride=1)
        self.up2_stage1 = nn.Sequential(
            *fish_block(args, n_feats=args.n_feats, multi_scale=args.multi, CSCR=args.CSCR, CSCR1=args.CSCR1,
                        spatial=args.spatial, CA=args.CA, SA=args.SA, PCSR=args.PCSR, PCSR1=args.PCSR1))
        self.up22 = nn.Sequential(*common.Upsampler(conv, 2, n_feats, act=False))  # n_feats*3
        if self.args.concat_tail2:
            self.concat_conv2 = nn.Conv2d(in_channels=n_feats + 3, out_channels=n_feats, kernel_size=1, stride=1)
        self.up2_stage2 = nn.Sequential(
            *fish_block(args, n_feats=args.n_feats, multi_scale=args.multi, CSCR=args.CSCR, CSCR1=args.CSCR1,
                        spatial=args.spatial, CA=args.CA, SA=args.SA, PCSR=args.PCSR, PCSR1=args.PCSR1))
        self.up23 = nn.Sequential(*common.Upsampler(conv, 2, n_feats, act=False))
        self.conv_tail3 = nn.Conv2d(in_channels=n_feats * 2, out_channels=n_feats, kernel_size=1, stride=1)
        if self.args.concat_tail3:
            self.concat_conv3 = nn.Conv2d(in_channels=n_feats + 3, out_channels=n_feats, kernel_size=1, stride=1)
        self.up2_stage3 = nn.Sequential(
            *fish_block(args, n_feats=args.n_feats, multi_scale=args.multi, CSCR=args.CSCR, CSCR1=args.CSCR1,
                        spatial=args.spatial, CA=args.CA, SA=args.SA, PCSR=args.PCSR, PCSR1=args.PCSR1))

        # define tail module
        m_tail = [
            # common.Upsampler(conv, scale, n_feats, act=False),
            conv(n_feats, args.n_colors, kernel_size)
        ]

        self.reduc = common.channelReduction()
        self.head = nn.Sequential(*m_head)
        self.tail = nn.Sequential(*m_tail)

    def forward(self, x, parsing=None):
        intp = x
        # print(parsing.shape)
        if parsing is not None:
            if self.args.large_parsing:
                p4 = F.interpolate(parsing, scale_factor=0.5, mode='nearest')
                p2 = F.interpolate(parsing, scale_factor=0.25, mode='nearest')
                p8 = parsing
                parsing = F.interpolate(parsing, scale_factor=0.125, mode='nearest')
            else:
                p2 = F.interpolate(parsing, scale_factor=2, mode='nearest')
                p4 = F.interpolate(parsing, scale_factor=4, mode='nearest')
                p8 = F.interpolate(parsing, scale_factor=8, mode='nearest')
        # for i in range(len(parsing_list)):
        #     print(i, parsing_list[i].shape)
        x = self.head(intp)
        # print(x.shape)
        x1 = self.up1(x)
        if self.args.concat_head1 and parsing is not None:
            x1 = torch.cat((x1, p2), 1)
            x1 = self.concat_conv1(x1)
        if self.args.PCSR or self.args.PCSR1:
            x = self.up_stage1[0](x1, p2)
            x = self.up_stage1[1](x, p2)
        else:
            x = self.up_stage1(x1)
        x2 = self.up2(x)
        if self.args.concat_head2 and parsing is not None:
            x2 = torch.cat((x2, p4), 1)
            x2 = self.concat_conv2(x2)
        if self.args.PCSR or self.args.PCSR1:
            x = self.up_stage2[0](x2, p4)
            x = self.up_stage2[1](x, p4)
        else:
            x = self.up_stage2(x2)
        x3 = self.up3(x)
        if self.args.concat_head3 and parsing is not None:
            x3 = torch.cat((x3, p8), 1)
            x3 = self.concat_conv3(x3)
        if self.args.PCSR or self.args.PCSR1:
            res1 = self.up_stage3[0](x3, p8)
            res1 = self.up_stage3[1](res1, p8)
        else:
            res1 = self.up_stage3(x3)
        # if self.args.shift_mean:
        #     res1 = self.add_mean(res1)
        if self.args.refine:
            inp = self.refine[0](x3, res1)
        elif self.args.refine1:
            inp = torch.cat((x3, res1), 1)
            res = self.refine1[0](inp)
            inp = self.reduc(inp)
            inp = inp + res
        elif self.args.refine2:
            inp = self.refine2[0](x1, x2, x3, res1)
        else:
            inp = torch.cat((x3, res1), 1)
            inp = self.reduc(inp)
        x4 = self.down1(inp)

        if self.args.concat_body1 and parsing is not None:
            x4 = torch.cat((x4, p4), 1)
            x4 = self.concat_conv3(x4)
        if self.args.PCSR or self.args.PCSR1:
            x = self.down_stage1[0](x4, p4)
            x = self.down_stage1[1](x, p4)
        else:
            x = self.down_stage1(x4)
        if self.args.refine:
            inp1 = self.refine[1](x2, x)
        elif self.args.refine1:
            inp = torch.cat((x, x2), 1)
            res = self.refine1[1](inp)
            inp = self.reduc(inp)
            inp1 = inp + res
        elif self.args.refine2:
            inp1 = self.refine2[1](x1, x2, x3, x)
        else:
            inp1 = torch.cat((x, x2), 1)
            inp1 = self.reduc(inp1)

        x5 = self.down2(inp1)
        if self.args.concat_body2 and parsing is not None:
            x5 = torch.cat((x5, p2), 1)
            x5 = self.concat_conv2(x5)
        if self.args.PCSR or self.args.PCSR1:
            x = self.down_stage2[0](x5, p2)
            x = self.down_stage2[1](x, p2)
        else:
            x = self.down_stage2(x5)
        if self.args.refine:
            inp2 = self.refine[2](x1, x)
        elif self.args.refine1:
            inp = torch.cat((x, x1), 1)
            res = self.refine1[2](inp)
            inp = self.reduc(inp)
            inp2 = inp + res
        elif self.args.refine2:
            inp2 = self.refine2[2](x1, x2, x3, x)
        else:
            inp2 = torch.cat((x, x1), 1)
            inp2 = self.reduc(inp2)

        x6 = self.down3(inp2)
        if self.args.concat_body3 and parsing is not None:
            x6 = torch.cat((x6, parsing), 1)
            x6 = self.concat_conv1(x6)
        if self.args.PCSR or self.args.PCSR1:
            x = self.down_stage3[0](x6, parsing)
            x = self.down_stage3[1](x, parsing)
        else:
            x = self.down_stage3(x6)
        if self.args.refine:
            inp3 = self.refine[3](x6, x)
        elif self.args.refine1:
            inp = torch.cat((x, x6), 1)
            res = self.refine1[3](inp)
            inp = self.reduc(inp)
            inp3 = inp + res
        elif self.args.refine2:
            inp3 = self.refine2[3](x6, x5, x4, x)
        else:
            inp3 = torch.cat((x, x6), 1)
            inp3 = self.conv_tail1(inp3)

        inp3 = self.conv(inp3)

        x = self.up21(inp3)
        # print(x.shape)
        if self.args.concat_tail1 and parsing is not None:
            x = torch.cat((x, p2), 1)
            x = self.concat_conv1(x)
        if self.args.PCSR or self.args.PCSR1:
            x = self.up2_stage1[0](x, p2)
            x = self.up2_stage1[1](x, p2)
        else:
            x = self.up2_stage1(x)
        if self.args.refine:
            inp4 = self.refine[4](x5, x)
        elif self.args.refine1:
            inp = torch.cat((x, x5), 1)
            res = self.refine1[4](inp)
            inp = self.reduc(inp)
            inp4 = inp + res
        elif self.args.refine2:
            inp4 = self.refine2[4](x6, x5, x4, x)
        else:
            inp4 = torch.cat((x, x5), 1)
            inp4 = self.conv_tail2(inp4)

        x = self.up22(inp4)
        if self.args.concat_tail2 and parsing is not None:
            x = torch.cat((x, p4), 1)
            x = self.concat_conv2(x)
        if self.args.PCSR or self.args.PCSR1:
            x = self.up2_stage2[0](x, p4)
            x = self.up2_stage2[1](x, p4)
        else:
            x = self.up2_stage2(x)
        if self.args.refine:
            inp5 = self.refine[5](x4, x)
        elif self.args.refine1:
            inp = torch.cat((x, x4), 1)
            res = self.refine1[5](inp)
            inp = self.reduc(inp)
            inp5 = inp + res
        elif self.args.refine2:
            inp5 = self.refine2[5](x6, x5, x4, x)
        else:
            inp5 = torch.cat((x, x4), 1)
            inp5 = self.conv_tail3(inp5)
        x = self.up23(inp5)
        if self.args.concat_tail3 and parsing is not None:
            x = torch.cat((x, p8), 1)
            x = self.concat_conv3(x)
        if self.args.PCSR or self.args.PCSR1:
            res = self.up2_stage3[0](x, p8)
            res = self.up2_stage3[0](res, p8)
        else:
            res = self.up2_stage3(x)
        x = self.tail(res)

        return x

    # def load_state_dict(self, state_dict, strict=True):
    #     own_state = self.state_dict()
    #     for name, param in state_dict.items():
    #         if name in own_state:
    #             if isinstance(param, nn.Parameter):
    #                 param = param.data
    #             try:
    #                 own_state[name].copy_(param)
    #             except Exception:
    #                 if name.find('tail') == -1:
    #                     raise RuntimeError('While copying the parameter named {}, '
    #                                        'whose dimensions in the model are {} and '
    #                                        'whose dimensions in the checkpoint are {}.'
    #                                        .format(name, own_state[name].size(), param.size()))
    #         elif strict:
    #             if name.find('tail') == -1:
    #                 raise KeyError('unexpected key "{}" in state_dict'
    #                                .format(name))


class LD(nn.Module):
    def __init__(self, args, conv=common.default_conv):
        super(LD, self).__init__()

        n_resblocks = 4
        n_feats = 64
        kernel_size = 3
        act = nn.ReLU(True)
        self.args = args
        # self.sub_mean = common.MeanShift(args.rgb_range)
        # self.add_mean = common.MeanShift(args.rgb_range, sign=1)
        m_head = [conv(args.n_colors, n_feats, kernel_size)]
        # define body module
        m_body = [
            common.ResBlock(
                conv, n_feats, kernel_size, act=act, res_scale=args.res_scale
            ) for _ in range(n_resblocks)
        ]

        m_body.append(conv(n_feats, n_feats, kernel_size))

        m_feature = [
            conv(n_feats, n_feats, kernel_size),
            conv(n_feats, 3, kernel_size)
        ]

        self.head = nn.Sequential(*m_head)
        self.body = nn.Sequential(*m_body)
        self.feature = nn.Sequential(*m_feature)

    def forward(self, x):
        # x = self.sub_mean(x)
        x = self.head(x)
        res = self.body(x)
        res += x
        feature = self.feature(res)
        # x = self.add_mean(x)
        return feature
