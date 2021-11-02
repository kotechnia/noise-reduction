import torch
import torch.nn as nn

from tscn.CME import GConv1d

class GatedDConv(nn.Module):  # Gated Transposed convolution
    def __init__(self, in_channels, dilation):
        super(GatedDConv, self).__init__()

        self.c1 = nn.Conv1d(in_channels=in_channels, out_channels=in_channels, kernel_size=3, padding=dilation,
                            dilation=dilation)
        self.c2 = nn.Conv1d(in_channels=in_channels, out_channels=in_channels, kernel_size=3, padding=dilation,
                            dilation=dilation)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()

    def forward(self, x):
        d1 = self.c1(x)

        d2 = self.c2(x)

        d2 = self.sigmoid(d2)
        output = d1 * d2

        return output


class CSRResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, M=5, r=1):
        super(CSRResBlock, self).__init__()

        self.c1 = nn.Conv1d(in_channels=in_channels, out_channels=64, kernel_size=1)
        self.pr1 = nn.PReLU()
        self.bn1 = nn.BatchNorm1d(num_features=64)

        self.c2 = nn.Conv1d(in_channels=128, out_channels=out_channels, kernel_size=1)

        self.d1 = GConv1d(in_channels=64, out_channels=64, dilation=r, padding=r)
        self.bn2 = nn.BatchNorm1d(num_features=64)
        self.pr2 = nn.PReLU()
        self.d2 = GConv1d(in_channels=64, out_channels=64, dilation=M, padding=M)
        self.bn3 = nn.BatchNorm1d(num_features=64)
        self.pr3 = nn.PReLU()

    def forward(self, x):
        x_ = x

        x = self.c1(x)
        x = self.bn1(x)
        x = self.pr1(x)

        d1 = self.d1(x)
        d2 = self.d2(x)

        x = torch.cat([d1, d2], dim=1)

        x = self.c2(x)

        output = x + x_

        return output


class CSRBlock(nn.Sequential):
    def __init__(self):
        super(CSRBlock, self).__init__(
            CSRResBlock(in_channels=256, out_channels=256, M=32, r=1),
            CSRResBlock(in_channels=256, out_channels=256, M=16, r=2),
            CSRResBlock(in_channels=256, out_channels=256, M=8, r=4),
            CSRResBlock(in_channels=256, out_channels=256, M=4, r=8),
            CSRResBlock(in_channels=256, out_channels=256, M=2, r=16),
            CSRResBlock(in_channels=256, out_channels=256, M=1, r=32),
        )


class CSRNet(nn.Module):
    def __init__(self, n_blocks):
        super(CSRNet, self).__init__()

        self.ec1 = nn.Conv2d(in_channels=4, out_channels=64, kernel_size=(2, 3), stride=(1, 2), padding=(0, 0))
        self.eg1 = nn.Conv2d(in_channels=4, out_channels=64, kernel_size=(2, 3), stride=(1, 2), padding=(0, 0))
        self.eb1 = nn.BatchNorm2d(num_features=64)
        self.ep1 = nn.PReLU()

        self.ec2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(2, 3), stride=(1, 2), padding=(0, 0))
        self.eg2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(2, 3), stride=(1, 2), padding=(0, 0))
        self.eb2 = nn.BatchNorm2d(num_features=64)
        self.ep2 = nn.PReLU()

        self.ec3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(2, 3), stride=(1, 2), padding=(0, 0))
        self.eg3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(2, 3), stride=(1, 2), padding=(0, 0))
        self.eb3 = nn.BatchNorm2d(num_features=64)
        self.ep3 = nn.PReLU()

        self.ec4 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(2, 3), stride=(1, 2), padding=(0, 0))
        self.eg4 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(2, 3), stride=(1, 2), padding=(0, 0))
        self.eb4 = nn.BatchNorm2d(num_features=64)
        self.ep4 = nn.PReLU()

        self.ec5 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(2, 3), stride=(1, 2), padding=(0, 0))
        self.eg5 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(2, 3), stride=(1, 2), padding=(0, 0))

        self.blocks = nn.Sequential(*[CSRBlock() for _ in range(n_blocks)])

        self.dc5 = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=(2, 3), stride=(1, 2),
                                      padding=(0, 0), output_padding=(0, 0))
        self.dg5 = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=(2, 3), stride=(1, 2),
                                      padding=(0, 0), output_padding=(0, 0))
        self.db5 = nn.BatchNorm2d(num_features=64)
        self.dp5 = nn.PReLU()

        self.dc4 = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=(2, 3), stride=(1, 2),
                                      padding=(0, 0), output_padding=(0, 0))
        self.dg4 = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=(2, 3), stride=(1, 2),
                                      padding=(0, 0), output_padding=(0, 0))
        self.db4 = nn.BatchNorm2d(num_features=64)
        self.dp4 = nn.PReLU()

        self.dc3 = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=(2, 3), stride=(1, 2),
                                      padding=(0, 0), output_padding=(0, 0))
        self.dg3 = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=(2, 3), stride=(1, 2),
                                      padding=(0, 0), output_padding=(0, 0))
        self.db3 = nn.BatchNorm2d(num_features=64)
        self.dp3 = nn.PReLU()

        self.dc2 = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=(2, 3), stride=(1, 2),
                                      padding=(0, 0), output_padding=(0, 1))
        self.dg2 = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=(2, 3), stride=(1, 2),
                                      padding=(0, 0), output_padding=(0, 1))
        self.db2 = nn.BatchNorm2d(num_features=64)
        self.dp2 = nn.PReLU()

        self.dc1 = nn.ConvTranspose2d(in_channels=128, out_channels=2, kernel_size=(2, 3), stride=(1, 2),
                                      padding=(0, 0), output_padding=(0, 0))
        self.dg1 = nn.ConvTranspose2d(in_channels=128, out_channels=2, kernel_size=(2, 3), stride=(1, 2),
                                      padding=(0, 0), output_padding=(0, 0))

        self.sigmoid = nn.Sigmoid()

    def forward(self, CCS, x):
        x = torch.cat([x, CCS], dim=1)
        x = x.permute(0, 1, 3, 2)

        e1 = self.ec1(x)
        e1g = self.eg1(x)
        x = e1 * self.sigmoid(e1g)
        x = self.eb1(x)
        x = self.ep1(x)

        e2 = self.ec2(x)
        e2g = self.eg2(x)
        x = e2 * self.sigmoid(e2g)
        x = self.eb2(x)
        x = self.ep2(x)

        e3 = self.ec3(x)
        e3g = self.eg3(x)
        x = e3 * self.sigmoid(e3g)
        x = self.eb3(x)
        x = self.ep3(x)

        e4 = self.ec4(x)
        e4g = self.eg4(x)
        x = e4 * self.sigmoid(e4g)
        x = self.eb4(x)
        x = self.ep4(x)

        e5 = self.ec5(x)
        e5g = self.eg5(x)
        x = e5 * self.sigmoid(e5g)

        x = x.permute(0, 1, 3, 2)
        x = torch.reshape(x, shape=(x.shape[0], 256, -1))
        x = self.blocks(x)

        x = torch.reshape(x, shape=(x.shape[0], 64, 4, -1))
        x = x.permute(0, 1, 3, 2)

        x_org = torch.cat([x, e5], dim=1)
        x_gate = torch.cat([x, e5g], dim=1)
        x = self.dc5(x_org)
        x_gate = self.dg5(x_gate)
        x = x * self.sigmoid(x_gate)
        x = self.db5(x)
        x = self.dp5(x)

        x_org = torch.cat([x, e4], dim=1)
        x_gate = torch.cat([x, e4g], dim=1)
        x = self.dc4(x_org)
        x_gate = self.dg4(x_gate)
        x = x * self.sigmoid(x_gate)
        x = self.db4(x)
        x = self.dp4(x)

        x_org = torch.cat([x, e3], dim=1)
        x_gate = torch.cat([x, e3g], dim=1)
        x = self.dc3(x_org)
        x_gate = self.dg3(x_gate)
        x = x * self.sigmoid(x_gate)
        x = self.db3(x)
        x = self.dp3(x)

        x_org = torch.cat([x, e2], dim=1)
        x_gate = torch.cat([x, e2g], dim=1)
        x = self.dc2(x_org)
        x_gate = self.dg2(x_gate)
        x = x * self.sigmoid(x_gate)
        x = self.db2(x)
        x = self.dp2(x)

        x_org = torch.cat([x, e1], dim=1)
        x_gate = torch.cat([x, e1g], dim=1)
        x = self.dc1(x_org)
        x_gate = self.dg1(x_gate)
        x = x * self.sigmoid(x_gate)

        x = x.permute(0, 1, 3, 2)
        x = torch.squeeze(x)

        return x
