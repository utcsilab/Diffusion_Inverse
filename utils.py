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
from tqdm import tqdm
# import sigpy as sp
def psnr(gt, est, max_pixel): 
    mse = np.mean((gt - est) ** 2) 
    max_pixel = max_pixel
    psnr = 20 * np.log10(max_pixel / np.sqrt(mse)) 
    return psnr

def nrmse_np(x,y):
    num = np.linalg.norm(x-y)
    denom = np.linalg.norm(x)
    return num/denom

def nrmse(x, y):
    num = torch.norm(x-y, p=2)
    denom = torch.norm(x,p=2)
    return num/denom
    return x

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

def power_iteration(A_normal,input_shape,iter,device,dtype):
    b = torch.randn(input_shape,dtype=dtype).to(device)
    max_eig_track = []
    for i in tqdm(range(iter)):
        b = A_normal(b)
        b = b/torch.norm(b,p=2)
        
        v = A_normal(b)
        max_eig_track.append(torch.norm(v,p=2)/torch.norm(b,p=2))

    max_eigen = max_eig_track[-1]
    min_eig_track = []
    b = torch.randn(input_shape,dtype=dtype).to(device)
    for i in tqdm(range(iter)):
        b = A_normal(b) - b*max_eigen
        b = b/torch.norm(b,p=2)
        
        v = A_normal(b)
        min_eig_track.append(torch.norm(v,p=2)/torch.norm(b,p=2))

    min_eigen = min_eig_track[-1]

    condition_num = max_eigen/min_eigen
    return max_eigen, min_eigen, condition_num