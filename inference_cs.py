# boiler plate imports
import numpy as np
import glob
import torch
from tqdm import tqdm
# import sigpy as sp
import matplotlib.pyplot as plt
import os
import argparse
from utils import nrmse, fft
from sampling_funcs import StackedRandomGenerator, simple_ODE_ps, general_forward_SDE_ps
import pickle
import dnnlib
from torch_utils import distributed as dist
from skimage.metrics import structural_similarity as ssim
from forwards import CS_utils
# dist.init()


parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--l_ss', type=float, default=1)
parser.add_argument('--sigma_max', type=float, default=10)
parser.add_argument('--num_steps', type=int, default=300)
parser.add_argument('--R', type=int, default=1)
parser.add_argument('--seed', type=int, default=100)
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


# load sample 
data_file = 'test_cat.pt'
gt_img = torch.load(data_file)


results_dir = './cs_results/'
if not os.path.exists(results_dir):
    os.makedirs(results_dir)

# load network
net_save = '/home/blevac/Diffusion_Inverse/models/edm-afhqv2-64x64-uncond-ve.pkl'

if dist.get_rank() != 0:
    torch.distributed.barrier()

# Load network.
dist.print0(f'Loading network from "{net_save}"...')
with dnnlib.util.open_url(net_save, verbose=(dist.get_rank() == 0)) as f:
    net = pickle.load(f)['ema'].to(device)



# Pick latents and labels.
batch_size=1
rnd = StackedRandomGenerator(device, [args.seed])
latents = rnd.randn([batch_size, net.img_channels, net.img_resolution, net.img_resolution], device=device)
class_labels = None


H = gt_img.shape[-2]
W = gt_img.shape[-1]
m = H*W//args.R
cs_utils = CS_utils(m=m,H=H,W=W)
A_forw = cs_utils.forward
meas = A_forw(gt_img)

recon_img = general_forward_SDE_ps(y=meas, A_forw=A_forw, task='', l_ss=args.l_ss, 
    net=net, latents=latents, class_labels=None, randn_like=torch.randn_like,
    num_steps=args.num_steps, sigma_min=0.002, sigma_max=args.sigma_max, rho=7,
    solver=args.solver, discretization=args.discretization, schedule='linear', scaling=args.scaling,
    epsilon_s=1e-3, C_1=0.001, C_2=0.008, M=1000, alpha=1,
    S_churn=0, S_min=0, S_max=float('inf'), S_noise=1, gt_img=gt_img, verbose = True)


img_nrmse = nrmse(gt_img, recon_img).item()

# cplx_recon=cplx_recon.detach().cpu().numpy()
# gt_img=gt_img.cpu().numpy()
# img_SSIM = ssim(abs(gt_img[0,0]), abs(cplx_recon[0,0]), data_range=abs(gt_img[0,0]).max() - abs(gt_img[0,0]).min())


# print('Sample %d, Seed %d, NRMSE: %.3f, SSIM: %.3f'%(args.sample,args.seed, img_nrmse, img_SSIM))

dict = { 
        'gt_img': gt_img,
        'recon':recon_img,
        'meas':meas,
        'forward_utils':cs_utils,
        # 'img_stack': img_stack,
        'nrmse':img_nrmse,
        # 'ssim': img_SSIM 
}

torch.save(dict, results_dir + '/checkpoint.pt')


