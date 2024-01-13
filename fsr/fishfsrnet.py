import common
import torch.nn.functional as F
import torch.nn as nn
import torch


def fish_block(args, conv=common.default_conv, n_feats=64):
    kernel_size = 3
    res = []
    act = nn.ReLU(True)


    res.append(common.PCSR1(
            conv, n_feats, kernel_size, act=act, res_scale=args.res_scale
        ))
    res.append(common.PCSR1(
            conv, n_feats, kernel_size, act=act, res_scale=args.res_scale
        ))

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

        # define head module
        m_head = [conv(args.n_colors, n_feats, kernel_size)]
        self.refine2 = nn.Sequential(*[common.Multi_scale_fusion_block(n_feats, scale=8),
                                          common.Multi_scale_fusion_block(n_feats, scale=4),
                                          common.Multi_scale_fusion_block(n_feats, scale=2),
                                          common.Multi_scale_fusion_block(n_feats, scale=2),
                                          common.Multi_scale_fusion_block(n_feats, scale=4),
                                          common.Multi_scale_fusion_block(n_feats, scale=8),
                                          ])
        # define body module

        self.up1 = nn.Sequential(*common.Upsampler(conv, 2, n_feats, act=False))

        self.up_stage1 = nn.Sequential(
            *fish_block(args))
        self.up2 = nn.Sequential(*common.Upsampler(conv, 2, n_feats, act=False))

        self.up_stage2 = nn.Sequential(
            *fish_block(args))
        self.up3 = nn.Sequential(*common.Upsampler(conv, 2, n_feats, act=False))

        self.up_stage3 = nn.Sequential(
            *fish_block(args))

        self.down1 = nn.Sequential(*common.invUpsampler(conv, 2, n_feats, act=False))

        self.down_stage1 = nn.Sequential(
            *fish_block(args))
        self.down2 = nn.Sequential(*common.invUpsampler(conv, 2, n_feats, act=False))

        self.down_stage2 = nn.Sequential(
            *fish_block(args))
        self.down3 = nn.Sequential(*common.invUpsampler(conv, 2, n_feats, act=False))

        self.down_stage3 = nn.Sequential(
            *fish_block(args))

        self.conv_tail1 = nn.Conv2d(in_channels=n_feats * 2, out_channels=n_feats, kernel_size=1, stride=1)
        self.conv = conv(n_feats, n_feats, 3)
        self.up21 = nn.Sequential(*common.Upsampler(conv, 2, n_feats, act=False))

        self.conv_tail2 = nn.Conv2d(in_channels=n_feats * 2, out_channels=n_feats, kernel_size=1, stride=1)
        self.up2_stage1 = nn.Sequential(
            *fish_block(args))
        self.up22 = nn.Sequential(*common.Upsampler(conv, 2, n_feats, act=False))  # n_feats*3

        self.up2_stage2 = nn.Sequential(
            *fish_block(args))
        self.up23 = nn.Sequential(*common.Upsampler(conv, 2, n_feats, act=False))
        self.conv_tail3 = nn.Conv2d(in_channels=n_feats * 2, out_channels=n_feats, kernel_size=1, stride=1)
        self.up2_stage3 = nn.Sequential(
            *fish_block(args))

        # define tail module
        m_tail = [

            conv(n_feats, args.n_colors, kernel_size)
        ]


        self.head = nn.Sequential(*m_head)
        self.tail = nn.Sequential(*m_tail)

    def forward(self, x, parsing=None):
        intp = x
        # print(parsing.shape)
        if parsing is not None:
            p2 = F.interpolate(parsing, scale_factor=2, mode='nearest')
            p4 = F.interpolate(parsing, scale_factor=4, mode='nearest')
            p8 = F.interpolate(parsing, scale_factor=8, mode='nearest')

        x = self.head(intp)
        # print(x.shape)
        x1 = self.up1(x)


        x = self.up_stage1[0](x1, p2)
        x = self.up_stage1[1](x, p2)

        x2 = self.up2(x)
        x = self.up_stage2[0](x2, p4)
        x = self.up_stage2[1](x, p4)

        x3 = self.up3(x)


        res1 = self.up_stage3[0](x3, p8)
        res1 = self.up_stage3[1](res1, p8)



        inp = self.refine2[0](x1, x2, x3, res1)

        x4 = self.down1(inp)


        x = self.down_stage1[0](x4, p4)
        x = self.down_stage1[1](x, p4)

        inp1 = self.refine2[1](x1, x2, x3, x)

        x5 = self.down2(inp1)


        x = self.down_stage2[0](x5, p2)
        x = self.down_stage2[1](x, p2)


        inp2 = self.refine2[2](x1, x2, x3, x)


        x6 = self.down3(inp2)

        x = self.down_stage3[0](x6, parsing)
        x = self.down_stage3[1](x, parsing)


        inp3 = self.refine2[3](x6, x5, x4, x)
        inp3 = self.conv(inp3)

        x = self.up21(inp3)

        x = self.up2_stage1[0](x, p2)
        x = self.up2_stage1[1](x, p2)


        inp4 = self.refine2[4](x6, x5, x4, x)


        x = self.up22(inp4)

        x = self.up2_stage2[0](x, p4)
        x = self.up2_stage2[1](x, p4)


        inp5 = self.refine2[5](x6, x5, x4, x)

        x = self.up23(inp5)

        res = self.up2_stage3[0](x, p8)
        res = self.up2_stage3[0](res, p8)

        x = self.tail(res)

        return x


