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
parser.add_argument('--ACS_perc', type=float, default=0.04)
parser.add_argument('--R', type=int, default=4)
parser.add_argument('--sample', type=int, default=0)
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--contrast_recon', type=str, default='PD')
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


# load sample 
data_file = f'/csiNAS2/slow/brett/multicontrast_validation_data/fastMRI_knee/ACS_perc{args.ACS_perc}_R={args.R}/sample{args.sample}_R{args.R}.pt'
contents = torch.load(data_file)
# print(contents.keys())
if args.contrast_recon == 'PD':
    s_maps          = torch.tensor(contents['maps_1']).cuda() # shape: [1,C,H,W]
    ksp           = torch.tensor(contents['ksp_1']).cuda()# shape: [1,C,H,W]
    mask          = torch.tensor(contents['mask_1']).cuda() # shape: [1,1,H,W]
    gt_img        = torch.tensor(contents['gt_img_1']).cuda() # shape [1,1,H,W]
    norm_c        = torch.tensor(contents['norm1']).cuda() # scalar
    class_idx = None

# normalize undersampled kspace and ground truth image
ksp = ksp/norm_c
gt_img = gt_img/norm_c

# setup MRI forward model + utilities
mri_utils = MRI_utils(maps=s_maps,mask=mask)
A_forw = mri_utils.forward

# designate + create save directory
results_dir = './mri_results/'
if not os.path.exists(results_dir):
    os.makedirs(results_dir)

# load network
if args.contrast_recon=='PD':
    net_save = '/csiNAS2/slow/brett/edm_outputs/00034-fastmri_PD_knee_preprocessed_7_21_23-uncond-ddpmpp-edm-gpus3-batch15-fp32-PD_knee_7_21_23/network-snapshot-010000.pkl'
if dist.get_rank() != 0:
        torch.distributed.barrier()
dist.print0(f'Loading network from "{net_save}"...')
with dnnlib.util.open_url(net_save, verbose=(dist.get_rank() == 0)) as f:
    net = pickle.load(f)['ema'].to(device)



# Pick latents and labels.
batch_size = 1
rnd = StackedRandomGenerator(device, [args.seed])
latents = rnd.randn([batch_size, net.img_channels, net.img_resolution, net.img_resolution], device=device)
class_labels = None



image_recon = general_forward_SDE_ps(y=ksp, A_forw=A_forw, task='mri', l_ss=args.l_ss, 
    net=net, latents=latents, class_labels=None, randn_like=torch.randn_like,
    num_steps=args.num_steps, sigma_min=0.002, sigma_max=args.sigma_max, rho=7,
    solver=args.solver, discretization=args.discretization, schedule='linear', scaling=args.scaling,
    epsilon_s=1e-3, C_1=0.001, C_2=0.008, M=1000, alpha=1,
    S_churn=0, S_min=0, S_max=float('inf'), S_noise=1, gt_img=gt_img, verbose = True)


cplx_recon = torch.view_as_complex(image_recon.permute(0,-2,-1,1).contiguous())[None] #shape: [1,1,H,W]
#unnormalize
cplx_recon = cplx_recon*norm_c
gt_img = gt_img*norm_c

img_nrmse = nrmse(abs(gt_img), abs(cplx_recon)).item()

cplx_recon=cplx_recon.detach().cpu().numpy()
gt_img=gt_img.cpu().numpy()
img_SSIM = ssim(abs(gt_img[0,0]), abs(cplx_recon[0,0]), data_range=abs(gt_img[0,0]).max() - abs(gt_img[0,0]).min())


print('Sample %d, Seed %d, NRMSE: %.3f, SSIM: %.3f'%(args.sample,args.seed, img_nrmse, img_SSIM))

dict = { 
        'gt_img': gt_img,
        'recon':cplx_recon,
        # 'img_stack': img_stack,
        'nrmse':img_nrmse,
        'ssim': img_SSIM 
}

torch.save(dict, results_dir + '/checkpoint.pt')


