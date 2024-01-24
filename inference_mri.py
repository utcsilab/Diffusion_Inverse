# boiler plate imports
import numpy as np
import glob
import torch
from tqdm import tqdm
# import sigpy as sp
import matplotlib.pyplot as plt
import os
import argparse
from utils import nrmse
from sampling_funcs import StackedRandomGenerator, general_forward_SDE_ps
import pickle
import dnnlib
from torch_utils import distributed as dist
from skimage.metrics import structural_similarity as ssim
from forwards import MRI_utils
# dist.init()


parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--l_ss', type=float, default=1)
parser.add_argument('--sigma_max', type=float, default=10)
parser.add_argument('--num_steps', type=int, default=300)
parser.add_argument('--R', type=int, default=4)
parser.add_argument('--seed', type=int, default=0)
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
data_file = './test_data/sample0_256.pt'
# data_file = '/home/blevac/Diffusion_Inverse/test_data/real_data_256x256.pt'
data_cont = torch.load(data_file)
kspace = data_cont['kspace'][0][None].cuda()
s_maps = data_cont['maps'][0][None].cuda()
gt_img = data_cont['gt_img'][0][None].cuda()
# mask = data_cont['mask'][0][None].cuda()
# mask = torch.tensor(sampling_mask_gen(ACS_perc=0.03, R=args.R, img_sz=gt_img.shape[-1])).cuda()
mask = torch.zeros(1,1,gt_img.shape[-2],gt_img.shape[-1]).cuda()
mask[:,:,:,0::args.R]=1
scale = torch.max(abs(gt_img))
gt_img=gt_img/scale
ksp = kspace/scale * mask



# setup MRI forward model + utilities
mri_utils = MRI_utils(maps=s_maps,mask=mask)
A_forw = mri_utils.forward
adj_img = mri_utils.adjoint(ksp)

# designate + create save directory
results_dir = './mri_results/'
if not os.path.exists(results_dir):
    os.makedirs(results_dir)

# load network
net_save = '/csiNAS2/slow/brett/edm_outputs/00037-fastmri_brain_preprocessed_9_25_23-uncond-ddpmpp-edm-gpus4-batch40-fp32-aspect/network-snapshot-004000.pkl'
if dist.get_rank() != 0:
        torch.distributed.barrier()
dist.print0(f'Loading network from "{net_save}"...')
with dnnlib.util_old.open_url(net_save, verbose=(dist.get_rank() == 0)) as f:
    net = pickle.load(f)['ema'].to(device)



# Pick latents and labels.
rnd = StackedRandomGenerator(device, args.latent_seeds)
latents = rnd.randn([batch_size, net.img_channels, gt_img.shape[-2], gt_img.shape[-1]], device=device)
class_labels = None


image_recon = general_forward_SDE_ps(y=ksp, A_forw=A_forw, task='mri', l_type='DPS', l_ss=args.l_ss, 
    net=net, latents=latents, class_labels=None, randn_like=torch.randn_like,
    num_steps=args.num_steps, sigma_min=0.002, sigma_max=args.sigma_max, rho=7,
    solver=args.solver, discretization=args.discretization, schedule='linear', scaling=args.scaling,
    epsilon_s=1e-3, C_1=0.001, C_2=0.008, M=1000, alpha=1,
    S_churn=args.S_churn, S_min=0, S_max=float('inf'), S_noise=1, verbose = True)

print('Net out shape: ', image_recon.shape)
cplx_recon = torch.view_as_complex(image_recon.permute(0,-2,-1,1).contiguous())[:,None] #shape: [1,1,H,W]

img_nrmse = nrmse(abs(gt_img), abs(cplx_recon)).item()

cplx_recon=cplx_recon.detach().cpu().numpy()
gt_img=gt_img.cpu().numpy()
img_SSIM = ssim(abs(gt_img[0,0]), abs(cplx_recon[0,0]), data_range=abs(gt_img[0,0]).max() - abs(gt_img[0,0]).min())

print('cplx net out shape: ',cplx_recon.shape)
print('NRMSE: %.3f, SSIM: %.3f'%(img_nrmse, img_SSIM))

dict = { 
        'gt_img': gt_img,
        'recon':cplx_recon,
        'adj_img': adj_img.cpu().numpy(),
        'nrmse':img_nrmse,
        'ssim': img_SSIM 
}

torch.save(dict, results_dir + '/checkpoint.pt')


