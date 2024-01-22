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

        n_resblocks =8
        n_feats = 64
        kernel_size = 3
        scale = 8
        act = nn.ReLU(True)
        self.args = args


        # define head module
        m_head = [conv(args.n_colors, n_feats, kernel_size)]
        self.refine2 = nn.Sequential(*[common.Multi_scale_fusion_block16(n_feats, scale=16),
                                          common.Multi_scale_fusion_block16(n_feats, scale=8),
                                          common.Multi_scale_fusion_block16(n_feats, scale=4),
                                          common.Multi_scale_fusion_block16(n_feats, scale=2),
                                          common.Multi_scale_fusion_block16(n_feats, scale=2),
                                          common.Multi_scale_fusion_block16(n_feats, scale=4),
                                          common.Multi_scale_fusion_block16(n_feats, scale=8),
                                          common.Multi_scale_fusion_block16(n_feats, scale=16),
                                          ])
        # define body module

        self.up1 = nn.Sequential(*[common.Upsampler(2, n_feats)])

        self.up_stage1 = nn.Sequential(*fish_block(args))
        self.up2 = nn.Sequential(*[common.Upsampler(2, n_feats)])

        self.up_stage2 = nn.Sequential(*fish_block(args))
        self.up3 = nn.Sequential(*[common.Upsampler( 2, n_feats)])

        self.up_stage3 = nn.Sequential(*fish_block(args))
        self.up4 = nn.Sequential(*[common.Upsampler(2, n_feats)])

        self.up_stage4 = nn.Sequential(*fish_block(args))

        self.down4 = nn.Sequential(*[common.invUpsampler(conv, 2, n_feats, act=False)])

        self.down_stage4 = nn.Sequential(*fish_block(args))


        self.down1 = nn.Sequential(*[common.invUpsampler(conv, 2, n_feats, act=False)])

        self.down_stage1 = nn.Sequential(*fish_block(args))
        self.down2 = nn.Sequential(*[common.invUpsampler(conv, 2, n_feats, act=False)])

        self.down_stage2 = nn.Sequential(*fish_block(args))
        self.down3 = nn.Sequential(*[common.invUpsampler(conv, 2, n_feats, act=False)])

        self.down_stage3 = nn.Sequential(*fish_block(args))


        self.up21 = nn.Sequential(*[common.Upsampler(conv, 2, 128, act=False)])

        self.up2_stage1 = nn.Sequential(*fish_block(args))
        self.up22 = nn.Sequential(*[common.Upsampler(conv, 2, n_feats*3, act=False)])#n_feats*3

        self.up2_stage2 = nn.Sequential(*fish_block(args))
        self.up23 = nn.Sequential(*[common.Upsampler(conv, 2, n_feats*4, act=False)])


        self.up2_stage3 = nn.Sequential(*fish_block(args))
        self.up24 = nn.Sequential(*[common.Upsampler(conv, 2, n_feats*5, act=False)])

        self.up2_stage4 = nn.Sequential(*fish_block(args))


        # define tail module
        m_tail = [

            conv(n_feats*5, args.n_colors, kernel_size)
        ]

        self.head = nn.Sequential(*m_head)
        self.tail = nn.Sequential(*m_tail)

    def forward(self, x, parsing=None):
        intp = x

        if parsing is not None:
            p2 = F.interpolate(parsing, scale_factor=2, mode='nearest')
            p4 = F.interpolate(parsing, scale_factor=4, mode='nearest')
            p8 = F.interpolate(parsing, scale_factor=8, mode='nearest')
            p16 = F.interpolate(parsing, scale_factor=16, mode='nearest')

        x = self.head(intp)
        # print(x.shape)
        x1 = self.up1(x) # 16 X 16

        x = self.up_stage1[0](x1, p2)
        x = self.up_stage1[1](x, p2)


        x2 = self.up2(x) # 32 X 32
        x = self.up_stage2[0](x2, p4)
        x = self.up_stage2[1](x, p4)

        x3 = self.up3(x) # 64 X 64
        res1 = self.up_stage3[0](x3, p8)
        res1 = self.up_stage3[1](res1, p8)



        x_up4 = self.up4(res1) # 128 X 128
        res1 = self.up_stage4[0](x_up4, p16)
        res_up4 = self.up_stage4[1](res1, p16)

        inp_down4 = self.refine2[0](x1, x2, x3, x_up4, res_up4)

        x_down4 = self.down4(inp_down4)# 64 X 64
        x = self.down_stage4[0](x_down4, p8)
        res_down4 = self.down_stage4[1](x, p8)

        inp = self.refine2[1](x1, x2, x3, x_up4, res_down4)

        x4 = self.down1(inp)# 32 X 32
        x = self.down_stage1[0](x4, p4)
        x = self.down_stage1[1](x, p4)

        inp1 = self.refine2[2](x1, x2, x3, x_up4, x)

        x5 = self.down2(inp1)# 16 X 16
        x = self.down_stage2[0](x5, p2)
        x = self.down_stage2[1](x, p2)

        inp2 = self.refine2[3](x1, x2, x3, x_up4, x)

        x6 = self.down3(inp2)# 8 X 8
        x = self.down_stage3[0](x6, parsing)
        x = self.down_stage3[1](x, parsing)

        inp3 = self.refine2[4](x1, x2, x3, x_up4, x)


        x = self.up21(inp3)# 16 X 16
        x = self.up2_stage1[0](x, p2)
        x = self.up2_stage1[1](x, p2)
        inp4 = self.refine2[5](x6, x5, x4, x_down4, x)

        x = self.up22(inp4)# 32 X 32
        x = self.up2_stage2[0](x, p4)
        x = self.up2_stage2[1](x, p4)

        inp5 = self.refine2[6](x6, x5, x4, x_down4, x)

        x = self.up23(inp5)# 64 X 64
        res = self.up2_stage3[0](x, p8)
        res = self.up2_stage3[0](res, p8)

        inp6 = self.refine2[7](x6, x5, x4, x_down4, res)

        x = self.up24(inp6)# 128 X 128
        res = self.up2_stage4[0](x, p16)
        res = self.up2_stage4[0](res, p16)

        x = self.tail(res)


        return x
