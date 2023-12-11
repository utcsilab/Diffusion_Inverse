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

    def adjoint(self,y):
        # ksp shape:  [B,1,H,W]
        # maps shape: [B,C,H,W]
        # mask shape: [B,1,H,W]

        sampled_ksp = self.mask*y
        coil_imgs = ifft(sampled_ksp)
        img_out = torch.sum(torch.conj(self.maps)*coil_imgs,dim=1) #sum over coil dimension

        return img_out[:,None,...]

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
        # print('x shape: ', x.shape)
        x_vec = torch.reshape(x,(self.batch, self.chan, self.n))
        # print('A shape: ', self.A.shape)
        # print('x_vec shape:  ',x_vec.shape)
        sampled_meas = torch.matmul(x_vec, self.A)      
        #meas shape: [B,C,m]
        return sampled_meas

    def meas_forward(self,x):
        #image shape: [B,C,H,W]
        # print('x shape: ', x.shape)
        x_vec = torch.reshape(x,(1, self.chan, self.n))
        # print('A shape: ', self.A.shape)
        # print('x_vec shape:  ',x_vec.shape)
        sampled_meas = torch.matmul(x_vec, self.A)      
        #meas shape: [B,C,m]
        return sampled_meas

    def adjoint(self,y):
        #input shape: [B,C,m]

        adj_vector = torch.matmul(y, self.A.T)
        adj_img    = torch.reshape(adj_vector, (self.batch, self.chan, self.H, self.W))       
        return adj_img


class CS_blind_utils:
    def __init__(self, m, H, W, error_scale=0, batch=1, chan=3, device='cuda'):
        self.m = m
        self.H = H
        self.W = W
        self.n = H*W
        self.chan = chan
        self.batch = batch
        self.A = torch.randn(self.n, self.m,dtype=torch.double,device=device)/self.m**.5
        self.A_error = (torch.randn(self.n, self.m,dtype=torch.double,device=device)/self.m**.5) * error_scale

    def assumed_forward(self,x):
        #image shape: [B,C,H,W]
        # print('x shape: ', x.shape)
        x_vec = torch.reshape(x,(self.batch, self.chan, self.n))
        # print('A shape: ', self.A.shape)
        # print('x_vec shape:  ',x_vec.shape)
        sampled_meas = torch.matmul(x_vec, self.A)      
        #meas shape: [B,C,m]
        return sampled_meas

    def true_forward(self,x):
        #image shape: [B,C,H,W]
        # print('x shape: ', x.shape)
        x_vec = torch.reshape(x,(self.batch, self.chan, self.n))
        # print('A shape: ', self.A.shape)
        # print('x_vec shape:  ',x_vec.shape)
        sampled_meas = torch.matmul(x_vec, self.A) + torch.matmul(x_vec, self.A_error)      
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
        self.A = torch.zeros((1,1,self.H*self.W), dtype=torch.double,device=device)
        self.A[:,:,idx[0:m]] = 1
        self.A = self.A.reshape(1,1,self.H,self.W)
    def forward(self,x):
        #image shape: [B,C,H,W]
        sampled_meas = self.A*x   
        #meas shape: [B,C,H,W]
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
        psf_pad = self.psf
        x_pad = x
        x_fft = fft(x_pad,norm='backward')
        psf_fft = fft(psf_pad,norm='backward')

        meas_fft = psf_fft*x_fft
        meas_pad = ifft(meas_fft,norm='backward')  
        return self.crop(meas_pad)
