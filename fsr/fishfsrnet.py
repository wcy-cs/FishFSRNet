import common
import torch.nn.functional as F
import torch.nn as nn
import torch


def fish_block(args, conv=common.default_conv, n_feats=64, PCSR1=False):
    kernel_size = 3
    res = []
    act = nn.ReLU(True)


    if PCSR1:
        res.append(common.PCSR1(
            conv, n_feats, kernel_size, act=act, res_scale=args.res_scale
        ))
        res.append(common.PCSR1(
            conv, n_feats, kernel_size, act=act, res_scale=args.res_scale
        ))
    else:
        res.append(common.ResBlock(conv, n_feats, kernel_size, act=act, res_scale=args.res_scale))
        res.append(common.ResBlock(conv, n_feats, kernel_size, act=act, res_scale=args.res_scale))
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
        if args.refine2:
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
            *fish_block(args, n_feats=args.n_feats, PCSR1=args.PCSR1))
        self.up2 = nn.Sequential(*common.Upsampler(conv, 2, n_feats, act=False))

        self.up_stage2 = nn.Sequential(
            *fish_block(args, n_feats=args.n_feats, PCSR1=args.PCSR1))
        self.up3 = nn.Sequential(*common.Upsampler(conv, 2, n_feats, act=False))

        self.up_stage3 = nn.Sequential(
            *fish_block(args, n_feats=args.n_feats,  PCSR1=args.PCSR1))

        self.down1 = nn.Sequential(*common.invUpsampler(conv, 2, n_feats, act=False))

        self.down_stage1 = nn.Sequential(
            *fish_block(args, n_feats=args.n_feats, PCSR1=args.PCSR1))
        self.down2 = nn.Sequential(*common.invUpsampler(conv, 2, n_feats, act=False))

        self.down_stage2 = nn.Sequential(
            *fish_block(args, n_feats=args.n_feats, PCSR1=args.PCSR1))
        self.down3 = nn.Sequential(*common.invUpsampler(conv, 2, n_feats, act=False))

        self.down_stage3 = nn.Sequential(
            *fish_block(args, n_feats=args.n_feats,  PCSR1=args.PCSR1))

        self.conv_tail1 = nn.Conv2d(in_channels=n_feats * 2, out_channels=n_feats, kernel_size=1, stride=1)
        self.conv = conv(n_feats, n_feats, 3)
        self.up21 = nn.Sequential(*common.Upsampler(conv, 2, n_feats, act=False))

        self.conv_tail2 = nn.Conv2d(in_channels=n_feats * 2, out_channels=n_feats, kernel_size=1, stride=1)
        self.up2_stage1 = nn.Sequential(
            *fish_block(args, n_feats=args.n_feats, PCSR1=args.PCSR1))
        self.up22 = nn.Sequential(*common.Upsampler(conv, 2, n_feats, act=False))  # n_feats*3

        self.up2_stage2 = nn.Sequential(
            *fish_block(args, n_feats=args.n_feats, PCSR1=args.PCSR1))
        self.up23 = nn.Sequential(*common.Upsampler(conv, 2, n_feats, act=False))
        self.conv_tail3 = nn.Conv2d(in_channels=n_feats * 2, out_channels=n_feats, kernel_size=1, stride=1)
        self.up2_stage3 = nn.Sequential(
            *fish_block(args, n_feats=args.n_feats, PCSR1=args.PCSR1))

        # define tail module
        m_tail = [

            conv(n_feats, args.n_colors, kernel_size)
        ]

        self.reduc = common.channelReduction()
        self.head = nn.Sequential(*m_head)
        self.tail = nn.Sequential(*m_tail)

    def forward(self, x, parsing=None):
        intp = x
        # print(parsing.shape)
        if parsing is not None:
            p2 = F.interpolate(parsing, scale_factor=2, mode='nearest')
            p4 = F.interpolate(parsing, scale_factor=4, mode='nearest')
            p8 = F.interpolate(parsing, scale_factor=8, mode='nearest')
        # for i in range(len(parsing_list)):
        #     print(i, parsing_list[i].shape)
        x = self.head(intp)
        # print(x.shape)
        x1 = self.up1(x)

        if  self.args.PCSR1:
            x = self.up_stage1[0](x1, p2)
            x = self.up_stage1[1](x, p2)
        else:
            x = self.up_stage1(x1)
        x2 = self.up2(x)
        if self.args.PCSR1:
            x = self.up_stage2[0](x2, p4)
            x = self.up_stage2[1](x, p4)
        else:
            x = self.up_stage2(x2)
        x3 = self.up3(x)

        if self.args.PCSR1:
            res1 = self.up_stage3[0](x3, p8)
            res1 = self.up_stage3[1](res1, p8)
        else:
            res1 = self.up_stage3(x3)
        # if self.args.shift_mean:
        #     res1 = self.add_mean(res1)

        if self.args.refine2:
            inp = self.refine2[0](x1, x2, x3, res1)
        else:
            inp = torch.cat((x3, res1), 1)
            inp = self.reduc(inp)
        x4 = self.down1(inp)

        if self.args.PCSR1:
            x = self.down_stage1[0](x4, p4)
            x = self.down_stage1[1](x, p4)
        else:
            x = self.down_stage1(x4)

        if self.args.refine2:
            inp1 = self.refine2[1](x1, x2, x3, x)
        else:
            inp1 = torch.cat((x, x2), 1)
            inp1 = self.reduc(inp1)

        x5 = self.down2(inp1)

        if self.args.PCSR1:
            x = self.down_stage2[0](x5, p2)
            x = self.down_stage2[1](x, p2)
        else:
            x = self.down_stage2(x5)

        if self.args.refine2:
            inp2 = self.refine2[2](x1, x2, x3, x)
        else:
            inp2 = torch.cat((x, x1), 1)
            inp2 = self.reduc(inp2)

        x6 = self.down3(inp2)

        if  self.args.PCSR1:
            x = self.down_stage3[0](x6, parsing)
            x = self.down_stage3[1](x, parsing)
        else:
            x = self.down_stage3(x6)

        if self.args.refine2:
            inp3 = self.refine2[3](x6, x5, x4, x)
        else:
            inp3 = torch.cat((x, x6), 1)
            inp3 = self.conv_tail1(inp3)

        inp3 = self.conv(inp3)

        x = self.up21(inp3)

        if self.args.PCSR1:
            x = self.up2_stage1[0](x, p2)
            x = self.up2_stage1[1](x, p2)
        else:
            x = self.up2_stage1(x)

        if self.args.refine2:
            inp4 = self.refine2[4](x6, x5, x4, x)
        else:
            inp4 = torch.cat((x, x5), 1)
            inp4 = self.conv_tail2(inp4)

        x = self.up22(inp4)

        if self.args.PCSR1:
            x = self.up2_stage2[0](x, p4)
            x = self.up2_stage2[1](x, p4)
        else:
            x = self.up2_stage2(x)

        if self.args.refine2:
            inp5 = self.refine2[5](x6, x5, x4, x)
        else:
            inp5 = torch.cat((x, x4), 1)
            inp5 = self.conv_tail3(inp5)
        x = self.up23(inp5)

        if self.args.PCSR1:
            res = self.up2_stage3[0](x, p8)
            res = self.up2_stage3[0](res, p8)
        else:
            res = self.up2_stage3(x)
        x = self.tail(res)

        return x


