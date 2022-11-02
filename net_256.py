import os
import torch
import torch.nn as nn


class Generator(nn.Module):

    def __init__(self, nz, nc, ngf):

        super(Generator, self).__init__()

        self.nz = nz
        self.nc = nc
        self.ngf = ngf

        self.ct4 = nn.ConvTranspose2d(self.nz, 8 * self.ngf, 4, 1, 0, bias=False)
        self.bn4 = nn.BatchNorm2d(8 * self.ngf)
        self.ac4 = nn.ReLU(inplace = True)

        self.ct8 = nn.ConvTranspose2d(8 * self.ngf, 8 * self.ngf, 4, 2, 1, bias=False)
        self.bn8 = nn.BatchNorm2d(8 * self.ngf)
        self.ac8 = nn.ReLU(inplace = True)

        self.ct16 = nn.ConvTranspose2d(8 * self.ngf, 4 * self.ngf, 4, 2, 1, bias=False)
        self.bn16 = nn.BatchNorm2d(4 * self.ngf)
        self.ac16 = nn.ReLU(inplace = True)

        self.ct32 = nn.ConvTranspose2d(4 * self.ngf, 4 * self.ngf, 4, 2, 1, bias=False)
        self.bn32 = nn.BatchNorm2d(4 * self.ngf)
        self.ac32 = nn.ReLU(inplace = True)

        self.ct64 = nn.ConvTranspose2d(4 * self.ngf, 2 * self.ngf, 4, 2, 1, bias=False)
        self.bn64 = nn.BatchNorm2d(2 * self.ngf)
        self.ac64 = nn.ReLU(inplace = True)

        self.ct128 = nn.ConvTranspose2d(2 * self.ngf, self.ngf, 4, 2, 1, bias=False)
        self.bn128 = nn.BatchNorm2d(self.ngf)
        self.ac128 = nn.ReLU(inplace = True)

        self.ct256 = nn.ConvTranspose2d(self.ngf, self.nc, 4, 2, 1, bias=False)
        self.ac256 = nn.Tanh()

    def forward(self, input):

        s = self.ct4(input)
        s = self.bn4(s)
        s = self.ac4(s)

        s = self.ct8(s)
        s = self.bn8(s)
        s = self.ac8(s)

        s = self.ct16(s)
        s = self.bn16(s)
        s = self.ac16(s)

        s = self.ct32(s)
        s = self.bn32(s)
        s = self.ac32(s)

        s = self.ct64(s)
        s = self.bn64(s)
        s = self.ac64(s)

        s = self.ct128(s)
        s = self.bn128(s)
        s = self.ac128(s)

        s = self.ct256(s)
        s = self.ac256(s)

        return s



class Discriminator(nn.Module):

    def __init__(self, nc, ndf):

        super(Discriminator, self).__init__()

        self.nc = nc
        self.ndf = ndf

        self.ct128 = nn.Conv2d(self.nc, self.ndf, 4, 2, 1, bias=False)
        self.bn128 = nn.BatchNorm2d(self.ndf)
        self.ac128 = nn.LeakyReLU(0.2, inplace = True)

        self.ct64 = nn.Conv2d(self.ndf, 2 * self.ndf, 4, 2, 1, bias=False)
        self.bn64 = nn.BatchNorm2d(2 * self.ndf)
        self.ac64= nn.LeakyReLU(0.2, inplace = True)

        self.ct32 = nn.Conv2d(2 * self.ndf, 4 * self.ndf, 4, 2, 1, bias=False)
        self.bn32 = nn.BatchNorm2d(4 * self.ndf)
        self.ac32 = nn.LeakyReLU(0.2, inplace = True)

        self.ct16 = nn.Conv2d(4 * self.ndf, 4 * self.ndf, 4, 2, 1, bias=False)
        self.bn16 = nn.BatchNorm2d(4 * self.ndf)
        self.ac16 = nn.LeakyReLU(0.2, inplace = True)

        self.ct8 = nn.Conv2d(4 * self.ndf, 8 * self.ndf, 4, 2, 1, bias=False)
        self.bn8 = nn.BatchNorm2d(8 * self.ndf)
        self.ac8 = nn.LeakyReLU(0.2, inplace = True)

        self.ct4 = nn.Conv2d(8 * self.ndf, 8 * self.ndf, 4, 2, 1, bias=False)
        self.bn4 = nn.BatchNorm2d(8 * self.ndf)
        self.ac4 = nn.LeakyReLU(0.2, inplace = True)

        self.ct1 = nn.Conv2d(8 * self.ndf, 1, 4, 1, 0, bias=False)
        self.ac1 = nn.Sigmoid()

    def forward(self, input):

        s = self.ct128(input)
        s = self.bn128(s)
        s = self.ac128(s)

        s = self.ct64(s)
        s = self.bn64(s)
        s = self.ac64(s)

        s = self.ct32(s)
        s = self.bn32(s)
        s = self.ac32(s)

        s = self.ct16(s)
        s = self.bn16(s)
        s = self.ac16(s)

        s = self.ct8(s)
        s = self.bn8(s)
        s = self.ac8(s)

        s = self.ct4(s)
        s = self.bn4(s)
        s = self.ac4(s)

        s = self.ct1(s)
        s = self.ac1(s)

        return s


