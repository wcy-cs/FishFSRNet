import common
import torch.nn as nn

class ParsingNet(nn.Module):
    def __init__(self, args, conv=common.default_conv):
        super(ParsingNet, self).__init__()

        n_resblocks = 8
        n_feats = 64
        kernel_size = 3
        act = nn.ReLU(True)
        self.args = args
        m_head = [conv(args.n_colors, n_feats, kernel_size)]
        # define body module
        m_body = [
            common.ResBlock(
                conv, n_feats, kernel_size, act=act, res_scale=args.res_scale
            ) for _ in range(n_resblocks)
        ]

        m_feature = [

            conv(n_feats, 3, kernel_size)
        ]

        self.head = nn.Sequential(*m_head)
        self.body = nn.Sequential(*m_body)
        self.feature = nn.Sequential(*m_feature)

    def forward(self, x):
        x = self.head(x)
        res = self.body(x)
        feature = self.feature(res)
        return feature
