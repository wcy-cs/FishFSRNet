import common
import torch.nn.functional as F
import torch.nn as nn
import torch


def fish_block(args, conv=common.default_conv, n_feats=64 ):
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
        self.refine2 = nn.Sequential(*[
                                          common.Multi_scale_fusion_block4(n_feats, scale=4),
                                          common.Multi_scale_fusion_block4(n_feats, scale=2),
                                          common.Multi_scale_fusion_block4(n_feats, scale=2),
                                          common.Multi_scale_fusion_block4(n_feats, scale=4),

                                          ])
        # define body module

        self.up1 = common.Upsampler(2, n_featse)

        self.up_stage1 = nn.Sequential(*fish_block(args, ))
        self.up2 = common.Upsampler(2, n_feats)

        self.up_stage2 = nn.Sequential(*fish_block(args
                                                   ))

        self.down2 = common.invUpsampler(conv, 2, n_feats, act=False)

        self.down_stage2 = nn.Sequential(*fish_block(args
                                                     ))
        self.down3 = common.invUpsampler(conv, 2, n_feats, act=False)

        self.down_stage3 = nn.Sequential(*fish_block(args
                                                     ))


        self.up21 = common.Upsampler(2, n_feats)

        self.up2_stage1 = nn.Sequential(*fish_block(args
                                                    ))
        self.up22 = common.Upsampler( 2, n_feats)#n_feats*3

        self.up2_stage2 = nn.Sequential(*fish_block(args))


        # define tail module
        m_tail = [

            conv(n_feats, args.n_colors, kernel_size)
        ]

        self.head = nn.Sequential(*m_head)
        self.tail = nn.Sequential(*m_tail)

    def forward(self, x, parsing=None):
        intp = x
        # print(intp.shape)
        if parsing is not None:
            p2 = F.interpolate(parsing, scale_factor=2, mode='nearest')
            p4 = F.interpolate(parsing, scale_factor=4, mode='nearest')

        x = self.head(intp)

        x1 = self.up1(x)

        x = self.up_stage1[0](x1, p2)
        x = self.up_stage1[1](x, p2)

        # x = self.up_stage1(x1)
        x2 = self.up2(x)
        x = self.up_stage2[0](x2, p4)
        x = self.up_stage2[1](x, p4)

        inp1 = self.refine2[0](x1, x2, x)

        x5 = self.down2(inp1)

        x = self.down_stage2[0](x5, p2)
        x = self.down_stage2[1](x, p2)

        inp2 = self.refine2[1](x1, x2, x)


        x6 = self.down3(inp2)
        x = self.down_stage3[0](x6, parsing)
        x = self.down_stage3[1](x, parsing)
        inp3 = self.refine2[2](x6, x5, x)

        x = self.up21(inp3)
        x = self.up2_stage1[0](x, p2)
        x = self.up2_stage1[1](x, p2)


        inp4 = self.refine2[3](x6, x5, x)

        x = self.up22(inp4)
        x = self.up2_stage2[0](x, p4)
        x = self.up2_stage2[1](x, p4)


        x = self.tail(x)


        return x

