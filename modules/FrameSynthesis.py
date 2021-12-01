import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    def __init__(self, in_ch = 64, out_ch = 64, leakyslope=0.02):
        super(ResidualBlock, self).__init__()
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.leakyslope = leakyslope

        self.layers = nn.Sequential(
                nn.Conv2d(in_channels=in_ch, out_channels=128, kernel_size=3, stride=1, padding=1, bias=True),
                nn.LeakyReLU(negative_slope=leakyslope),
                nn.Conv2d(in_channels=128, out_channels=out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            )

    def forward(self, input):
        output = self.layers(input)
        output = output + input
        return output


class FrameSynthesis(nn.Module):
    def __init__(self, in_chans, out_chans=3):
        super(FrameSynthesis, self).__init__()
        self.in_chans = in_chans
        self.out_chans = out_chans

        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=in_chans, out_channels=64, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU()
        )
        self.layer2 = nn.Sequential(
            [ResidualBlock()] * 10
        )
        self.layer3 = nn.Conv2d(in_channels=64, out_channels=out_chans, kernel_size=3, stride=1, padding=1, bias=True)

    def forward(self, input):
        return self.layer3(self.layer2(self.layer1(input))) + input[:, :3] # Add only warped images
