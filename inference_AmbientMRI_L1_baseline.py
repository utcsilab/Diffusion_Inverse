# boiler plate imports
import numpy as np
import glob
import torch
from tqdm import tqdm
# import sigpy as sp
import matplotlib.pyplot as plt
import os
import argparse
from utils import nrmse_np, psnr
from sampling_funcs import StackedRandomGenerator, general_forward_SDE_ps
import pickle
import dnnlib
from torch_utils import distributed as dist
from skimage.metrics import structural_similarity as ssim
from forwards import MRI_utils
import json
from collections import OrderedDict
# dist.init()


parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--sample', type=int, default=0)
parser.add_argument('--l_ss', type=float, default=1)
parser.add_argument('--sigma_max', type=float, default=10)
parser.add_argument('--num_steps', type=int, default=300)
parser.add_argument('--R', type=int, default=4)
parser.add_argument('--train_R', type=int, default=4)
parser.add_argument('--seed', type=int, default=10)
parser.add_argument('--latent_seeds', type=int, nargs='+' ,default= 0)
parser.add_argument('--S_churn', type=float, default=40)
parser.add_argument('--net_arch', type=str, default='ddpmpp') 
parser.add_argument('--discretization', type=str, default='edm') # ['vp', 've', 'iddpm', 'edm']
parser.add_argument('--solver', type=str, default='euler') # ['euler', 'heun']
parser.add_argument('--schedule', type=str, default='vp') # ['vp', 've', 'linear']
parser.add_argument('--scaling', type=str, default='vp') # ['vp', 'none']

args   = parser.parse_args()

os.environ["CUDA_DEVICE_ORDER"]= "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]=str(args.gpu)
#seeds
torch.manual_seed(args.seed)
np.random.seed(args.seed)
device=torch.device('cuda')
batch_size=len(args.latent_seeds)

#load data and preprocess
data_file = '/csiNAS/asad/data/brain_fastMRI/val_samples_ambient/sample_%d.pt'%args.sample
cont = torch.load(data_file)
mask_str = 'mask_%d'%args.R

gt_img = cont['gt'][None,None,...].cuda() #shape [1,1,384,320]
s_maps = cont['s_map'][None,...].cuda() # shape [1,16,384,320]
fs_ksp = cont['ksp'][None,...].cuda() #shape [1,16,384,320]
mask = cont[mask_str][None, ...].cuda() # shape [1,1,384,320]
ksp = mask*fs_ksp
# ksp = ksp/np.percentile(abs(gt_img.cpu()),99)

# print('gt_img shape: ',gt_img.shape )
# print('s_maps shape: ',s_maps.shape )
# print('mask shape: ',mask.shape )
# print('ksp shape: ',ksp.shape )



# setup MRI forward model + utilities
mri_utils = MRI_utils(maps=s_maps,mask=mask)
A_forw = mri_utils.forward
adj_img = mri_utils.adjoint(ksp)

# designate + create save directory
results_dir = '/csiNAS2/slow/brett/ambientmri_L1_R=%d_baseline_DPS_results/sample%d/R=%d/'%(args.train_R, args.sample,args.R)
if not os.path.exists(results_dir):
    os.makedirs(results_dir)


# load network
# net_save = '/csiNAS2/slow/brett/edm_outputs/00037-fastmri_brain_preprocessed_9_25_23-uncond-ddpmpp-edm-gpus4-batch40-fp32-aspect/network-snapshot-004000.pkl'
# net_save = '/csiNAS/brett/ambient_4/00000-ksp_brainMRI_384-uncond-ddpmpp-ambient-gpus4-batch32-fp16/network-snapshot-015004.pkl'
net_save = '/csiNAS2/slow/sidharth/EDM_ambient/r=%d/network-snapshot-001200.pkl'%(args.train_R)
if dist.get_rank() != 0:
        torch.distributed.barrier()
dist.print0(f'Loading network from "{net_save}"...')
with dnnlib.util.open_url(net_save, verbose=(dist.get_rank() == 0)) as f:
    net = pickle.load(f)['ema'].to(device)


# Pick latents and labels.
rnd = StackedRandomGenerator(device, args.latent_seeds)
latents = rnd.randn([batch_size, 2, gt_img.shape[-2], gt_img.shape[-1]], device=device)
class_labels = None


image_recon = general_forward_SDE_ps(y=ksp, A_forw=A_forw, task='mri', l_type='DPS', l_ss=args.l_ss, 
    net=net, latents=latents, class_labels=None, randn_like=torch.randn_like,
    num_steps=args.num_steps, sigma_min=0.002, sigma_max=args.sigma_max, rho=7,
    solver=args.solver, discretization=args.discretization, schedule='linear', scaling=args.scaling,
    epsilon_s=1e-3, C_1=0.001, C_2=0.008, M=1000, alpha=1,
    S_churn=args.S_churn, S_min=0, S_max=float('inf'), S_noise=1, verbose = True)

cplx_recon = torch.view_as_complex(image_recon.permute(0,-2,-1,1).contiguous())[:,None] #shape: [1,1,H,W]


cplx_recon=cplx_recon.detach().cpu().numpy()
mean_recon=np.mean(cplx_recon,axis=0)[None]
gt_img=gt_img.cpu().numpy()
img_nrmse = nrmse_np(abs(gt_img[0,0]), abs(mean_recon[0,0]))
img_SSIM = ssim(abs(gt_img[0,0]), abs(mean_recon[0,0]), data_range=abs(gt_img[0,0]).max() - abs(gt_img[0,0]).min())
img_PSNR = psnr(gt=abs(gt_img[0,0]), est=abs(mean_recon[0]),max_pixel=np.amax(abs(gt_img)))

# print('cplx net out shape: ',cplx_recon.shape)
print('Sample %d, R: %d, NRMSE: %.3f, SSIM: %.3f, PSNR: %.3f'%(args.sample, args.R, img_nrmse, img_SSIM, img_PSNR))

dict = { 
        'gt_img': gt_img,
        'recon':cplx_recon,
        'adj_img': adj_img.cpu().numpy(),
        'nrmse':img_nrmse,
        'ssim': img_SSIM,
        'psnr': img_PSNR
}

torch.save(dict, results_dir + '/checkpoint.pt')


