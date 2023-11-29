#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  2 18:29:31 2021

@author: yanni
"""

import torch
import torch.fft as torch_fft
import torch.nn as nn

import numpy as np
# import sigpy as sp
import torchvision
import torchvision.transforms as transforms

# Centered, orthogonal ifft in torch >= 1.7
def ifft(x,norm = 'ortho'):
    x = torch_fft.ifftshift(x, dim=(-2, -1))
    x = torch_fft.ifft2(x, dim=(-2, -1), norm=norm)
    x = torch_fft.fftshift(x, dim=(-2, -1))
    return x

# Centered, orthogonal fft in torch >= 1.7
def fft(x,norm='ortho'):
    x = torch_fft.fftshift(x, dim=(-2, -1))
    x = torch_fft.fft2(x, dim=(-2, -1), norm=norm)
    x = torch_fft.ifftshift(x, dim=(-2, -1))
    return x


def mri_forward(image, maps, mask):
    #image shape: [B,1,H,W]
    #maps shape:  [B,C,H,W]
    # mask shape: [B,1,H,W]

    coil_imgs = maps*image
    coil_ksp = fft(coil_imgs)
    sampled_ksp = mask*coil_ksp
    return sampled_ksp

def mri_adjoint(ksp, maps, mask):
    # ksp shape:  [B,1,H,W]
    # maps shape: [B,C,H,W]
    # mask shape: [B,1,H,W]

    sampled_ksp = mask*ksp
    coil_imgs = ifft(sampled_ksp)
    img_out = torch.sum(torch.conj(maps)*coil_imgs,dim=1) #sum over coil dimension

    return img_out[:,None,...]

class MRI_utils:
    def __init__(self, mask, maps):
        self.mask = mask
        self.maps = maps

    def forward(self,x):
        #image shape: [B,1,H,W]
        #maps shape:  [B,C,H,W]
        # mask shape: [B,1,H,W]
        x_cplx = torch.view_as_complex(x.permute(0,-2,-1,1).contiguous())[None]
        coil_imgs = self.maps*x_cplx
        coil_ksp = fft(coil_imgs)
        sampled_ksp = self.mask*coil_ksp
        return sampled_ksp

class CS_utils:
    def __init__(self, m, H, W, batch=1, chan=3, device='cuda'):
        self.m = m
        self.H = H
        self.W = W
        self.n = H*W
        self.chan = chan
        self.batch = batch
        self.A = torch.randn(self.n, self.m,dtype=torch.double,device=device)/self.m**.5

    def forward(self,x):
        #image shape: [B,C,H,W]
        x_vec = torch.reshape(x,(self.batch, self.chan, self.n))
        sampled_meas = torch.matmul(x_vec, self.A)      
        #meas shape: [B,C,m]
        return sampled_meas

    def adjoint(self,y):
        #input shape: [B,C,m]

        adj_vector = torch.matmul(y, self.A.T)
        adj_img    = torch.reshape(adj_vector, (self.batch, self.chan, self.H, self.W))       
        return adj_img


class InPaint_utils:
    def __init__(self, m, H, W, batch=1, chan=3, device='cuda'):
        self.m = m
        self.H = H
        self.W = W
        self.chan = chan
        self.batch = batch
        idx = torch.arange(self.H*self.W)
        idx=np.random.permutation(idx)
        self.A = torch.zeros((self.batch,1,self.H*self.W), dtype=torch.double,device=device)
        self.A[:,:,idx[0:m]] = 1
        self.A = self.A.reshape(self.batch,1,self.H,self.W)
    def forward(self,x):
        #image shape: [B,C,H,W]
        sampled_meas = self.A*x   
        #meas shape: [B,C,m]
        return sampled_meas

    def adjoint(self,y):
        #input shape: [B,C,m]

        adj_img = self.A*y         
        return adj_img

class LSI_utils:
    def __init__(self, psf, H, W, device='cuda'):
        self.psf = psf
        self.H = H
        self.W = W

    def pad(self, x):
        # to pad W/2->left, H/2->top, W/2-> right, H/2-> bottom
        transform = transforms.Pad((self.W//2,self.H//2,self.W//2,self.H//2))
        x_pad = transform(x)
        return x_pad

    def crop(self,x):
        transform = transforms.CenterCrop((self.H,self.W))
        x_crop = transform(x)
        return x_crop

    def forward(self,x):
        #image shape: [B,C,H,W]
        psf_pad = self.pad(self.psf)
        x_pad   = self.pad(x)
        # psf_pad = self.psf
        # x_pad = x
        x_fft = fft(x_pad,norm='backward')
        psf_fft = fft(psf_pad,norm='backward')

        meas_fft = psf_fft*x_fft
        meas_pad = ifft(meas_fft,norm='backward')  
        return self.crop(meas_pad)

    def adjoint(self,y):
        #input shape: [B,C,m]

        adj_img = self.A*y         
        return adj_img

def gaussuian_filter(kernel_size, sigma=.05, muu=0):
 
    # Initializing value of x,y as grid of kernel size
    # in the range of kernel size
 
    x, y = np.meshgrid(np.linspace(-2, 2, kernel_size),
                       np.linspace(-2, 2, kernel_size))
    dst = np.sqrt(x**2+y**2)
 
    # lower normal part of gaussian
    normal = 1/(2*np.pi * sigma**2)**.5
 
    # Calculating Gaussian filter
    gauss = np.exp(-((dst-muu)**2 / (2.0 * sigma**2))) * normal
    gauss = gauss/np.sum(gauss)
    return gauss