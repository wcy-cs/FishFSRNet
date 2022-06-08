
import torch
import torch.nn as nn
import torch.nn.functional as F



class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class ChannelGate(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16, pool_types=None):
        super(ChannelGate, self).__init__()
        if pool_types is None:
            pool_types = ['avg', 'max']
        self.gate_channels = gate_channels
        self.mlp = nn.Sequential(
            Flatten(),
            nn.Linear(gate_channels, int(gate_channels // reduction_ratio)),
            nn.ReLU(),
            nn.Linear(int(gate_channels // reduction_ratio), gate_channels)
        )
        self.pool_types = pool_types

    def forward(self, x):
        channel_att_sum = None
        for pool_type in self.pool_types:
            if pool_type == 'avg':
                avg_pool = F.avg_pool2d(x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp(avg_pool)
            elif pool_type == 'max':
                max_pool = F.max_pool2d(x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp(max_pool)
            elif pool_type == 'lp':
                lp_pool = F.lp_pool2d(x, 2, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp(lp_pool)
            elif pool_type == 'var':
                var_pool = variance_pool(x)
                channel_att_raw = self.mlp(var_pool)
            elif pool_type == 'lse':
                # LSE pool only
                lse_pool = logsumexp_2d(x)
                channel_att_raw = self.mlp(lse_pool)

            if channel_att_sum is None:
                channel_att_sum = channel_att_raw
            else:
                channel_att_sum = channel_att_sum + channel_att_raw

        scale = torch.sigmoid(channel_att_sum).unsqueeze(2).unsqueeze(3).expand_as(x)
        return x * scale


def logsumexp_2d(tensor):
    tensor_flatten = tensor.view(tensor.size(0), tensor.size(1), -1)
    s, _ = torch.max(tensor_flatten, dim=2, keepdim=True)
    outputs = s + (tensor_flatten - s).exp().sum(dim=2, keepdim=True).log()
    return outputs


class ChannelPool(nn.Module):
    def forward(self, x):
        return torch.cat((torch.max(x, 1)[0].unsqueeze(1), torch.mean(x, 1).unsqueeze(1)), dim=1)



class CSAR_SpatialGate(nn.Module):
    def __init__(self, n_feats, gama=2):
        super(CSAR_SpatialGate, self).__init__()

        self.spatial_layer1 = nn.Conv2d(in_channels=n_feats, out_channels=gama * n_feats, kernel_size=1, stride=1)
        self.spatial_layer2 = nn.ReLU()
        self.spatial_layer3 = nn.Conv2d(in_channels=gama * n_feats, out_channels=n_feats, kernel_size=1, stride=1)

    def forward(self, x):
        x_compress = self.spatial_layer1(x)
        x_out = self.spatial_layer2(x_compress)
        x_out = self.spatial_layer3(x_out)
        scale = torch.sigmoid(x_out)  # broadcasting
        return x * scale



def variance_pool(x):
    my_mean = x.mean(dim=3, keepdim=True).mean(dim=2, keepdim=True)
    return (x - my_mean).pow(2).mean(dim=3, keepdim=False).mean(dim=2, keepdim=False).view(x.size()[0], x.size()[1], 1,
                                                                                           1)


