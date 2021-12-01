import torch
from torch import nn


class KernelEstimation(nn.Module):
    def __init__(self, in_chans=3, out_chans=16, leaky_slope=0.02):
        super(KernelEstimation, self).__init__()
        self.in_chans = in_chans
        self.out_chans = out_chans

        # Be careful, below codes use convolution 'using' padding
        # Did not use batchnorm: used layernorm (since differences are big for batches)
        # encoder layer 1
        self.enc_1_convrelu1 = nn.Sequential(
            nn.Conv2d(in_channels=in_chans, out_channels=64, kernel_size=3, stride=1, padding=1, bias=True),
            nn.LayerNorm(num_features=64),
            nn.LeakyReLU(leaky_slope, inplace=True)
        )
        self.enc_1_convrelu2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=True),
            nn.LayerNorm(num_features=64),
            nn.LeakyReLU(leaky_slope, inplace=True)
        )
        self.enc_1_pool = nn.MaxPool2d(kernel_size=2)

        # encoder layer 2
        self.enc_2_convrelu1 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1, bias=True),
            nn.LayerNorm(num_features=128),
            nn.LeakyReLU(leaky_slope, inplace=True)
        )
        self.enc_2_convrelu2 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1, bias=True),
            nn.LayerNorm(num_features=128),
            nn.LeakyReLU(leaky_slope, inplace=True)
        )
        self.enc_2_pool = nn.MaxPool2d(kernel_size=2)

        # endoder layer 3
        self.enc_3_convrelu1 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1, bias=True),
            nn.LayerNorm(num_features=256),
            nn.LeakyReLU(leaky_slope, inplace=True)
        )
        self.enc_3_convrelu2 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1, bias=True),
            nn.LayerNorm(num_features=256),
            nn.LeakyReLU(leaky_slope, inplace=True)
        )
        self.enc_3_pool = nn.MaxPool2d(kernel_size=2)

        # encoder layer 4
        self.enc_4_convrelu1 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1, bias=True),
            nn.LayerNorm(num_features=512),
            nn.LeakyReLU(leaky_slope, inplace=True)
        )
        self.enc_4_convrelu2 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1, bias=True),
            nn.LayerNorm(num_features=512),
            nn.LeakyReLU(leaky_slope, inplace=True)
        )
        self.enc_4_pool = nn.MaxPool2d(kernel_size=2)

        # encoder layer 5
        self.enc_5_convrelu1 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1, bias=True),
            nn.LayerNorm(num_features=512),
            nn.LeakyReLU(leaky_slope, inplace=True)
        )
        self.enc_5_convrelu2 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1, bias=True),
            nn.LayerNorm(num_features=512),
            nn.LeakyReLU(leaky_slope, inplace=True)
        )

        # decoder layer 5
        self.dec_5_upsample = nn.ConvTranspose2d(in_channels=512, out_channels=512, kernel_size=2, stride=2, padding=0,
                                                 bias=True)

        # decoder layer 4
        self.dec_4_convrelu2 = nn.Sequential(
            nn.Conv2d(in_channels=1024, out_channels=512, kernel_size=3, stride=1, padding=1, bias=True),
            nn.LayerNorm(num_features=512),
            nn.LeakyReLU(leaky_slope, inplace=True)
        )
        self.dec_4_convrelu1 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1, bias=True),
            nn.LayerNorm(num_features=512),
            nn.LeakyReLU(leaky_slope, inplace=True)
        )
        self.dec_4_upsample = nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=2, stride=2, padding=0,
                                                 bias=True)

        # decoder layer 3
        self.dec_3_convrelu2 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=256, kernel_size=3, stride=1, padding=1, bias=True),
            nn.LayerNorm(num_features=256),
            nn.LeakyReLU(leaky_slope, inplace=True)
        )
        self.dec_3_convrelu1 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1, bias=True),
            nn.LayerNorm(num_features=256),
            nn.LeakyReLU(leaky_slope, inplace=True)
        )
        self.dec_3_upsample = nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=2, stride=2, padding=0,
                                                 bias=True)

        # decoder layer 2
        self.dec_2_convrelu2 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=128, kernel_size=3, stride=1, padding=1, bias=True),
            nn.LayerNorm(num_features=128),
            nn.LeakyReLU(leaky_slope, inplace=True)
        )
        self.dec_2_convrelu1 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1, bias=True),
            nn.LayerNorm(num_features=128),
            nn.LeakyReLU(leaky_slope, inplace=True)
        )
        self.dec_2_upsample = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=2, stride=2, padding=0,
                                                 bias=True)

        # decoder layer 1
        self.dec_1_convrelu2 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, stride=1, padding=1, bias=True),
            nn.LayerNorm(num_features=64),
            nn.LeakyReLU(leaky_slope, inplace=True)
        )
        self.dec_1_convrelu1 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=True),
            nn.LayerNorm(num_features=64),
            nn.LeakyReLU(leaky_slope, inplace=True)
        )
        self.dec_1_final = nn.Conv2d(in_channels=64, out_channels=out_chans, kernel_size=1, stride=1, padding=0,
                                     bias=True)

    def norm(self, x):
        if len(x.shape) == 3:
            B, H, W = x.shape
            x = x.view(B, H*W)
            mean = x.mean(dim=1).view(B, 1, 1)
            std = x.std(dim=1).view(B, 1, 1)
            x = x.view(B, H, W)
            return (x - mean) / std, mean, std
        elif len(x.shape) == 4:
            B, Ch, H, W = x.shape
            x = x.view(B, Ch, H * W)
            mean = x.mean(dim=2).view(B, Ch, 1, 1)
            # print(mean.shape)
            std = x.std(dim=2).view(B, Ch, 1, 1)
            x = x.view(B, Ch, H, W)
            return (x - mean) / std, mean, std

    def unnorm(self, x, mean, std):
        return x * std + mean

    def forward(self, Input):

        # Normalize for spatial dimensions (not for batch, column dimensions)
        Input, mean, std = self.norm(Input)

        # encoder
        enc1 = self.enc_1_convrelu1(Input)
        enc1 = self.enc_1_convrelu2(enc1)
        enc2 = self.enc_1_pool(enc1)
        enc2 = self.enc_2_convrelu1(enc2)
        enc2 = self.enc_2_convrelu2(enc2)
        enc3 = self.enc_2_pool(enc2)
        enc3 = self.enc_3_convrelu1(enc3)
        enc3 = self.enc_3_convrelu2(enc3)
        enc4 = self.enc_3_pool(enc3)
        enc4 = self.enc_4_convrelu1(enc4)
        enc4 = self.enc_4_convrelu2(enc4)
        Input = self.enc_4_pool(enc4)
        Input = self.enc_5_convrelu1(Input)
        Input = self.enc_5_convrelu2(Input)

        # decoder
        Input = self.dec_5_upsample(Input)
        Input = torch.cat([enc4, Input], dim=1)
        Input = self.dec_4_convrelu2(Input)
        Input = self.dec_4_convrelu1(Input)

        Input = self.dec_4_upsample(Input)
        Input = torch.cat([enc3, Input], dim=1)
        Input = self.dec_3_convrelu2(Input)
        Input = self.dec_3_convrelu1(Input)

        Input = self.dec_3_upsample(Input)
        Input = torch.cat([enc2, Input], dim=1)
        Input = self.dec_2_convrelu2(Input)
        Input = self.dec_2_convrelu1(Input)

        Input = self.dec_2_upsample(Input)
        Input = torch.cat([enc1, Input], dim=1)
        Input = self.dec_1_convrelu2(Input)
        Input = self.dec_1_convrelu1(Input)
        Input = self.dec_1_final(Input)

        # unnorm output: NO NEED TO DO IT, SINCE WE ARE NOT RECONSTRUCTING IMAGE
        # Input = self.unnorm(Input, mean, std)

        return Input