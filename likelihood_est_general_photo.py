# boiler plate imports
import numpy as np
import glob
import torch
from tqdm import tqdm
# import sigpy as sp
import matplotlib.pyplot as plt
import os
import argparse
from utils import nrmse, gaussuian_filter
from sampling_funcs import StackedRandomGenerator, general_forward_SDE_ps
import pickle
import dnnlib
from torch_utils import distributed as dist
from skimage.metrics import structural_similarity as ssim
from forwards import LSI_utils, InPaint_utils, CS_utils
from opt import conjgrad, grad_descent
# dist.init()


parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--num_steps', type=int, default=300)
parser.add_argument('--l_ss', type=float, default=1)
parser.add_argument('--lam', type=float, default=0.000)
parser.add_argument('--R', type=int, default=4)
parser.add_argument('--proj', type=int, default=1) #[0,1]
parser.add_argument('--task', type=str, default='deconv') # ['deconv', 'inpaint', 'compsens']
parser.add_argument('--gen_seed', type=int ,default= 100)
parser.add_argument('--solver', type=str ,default= 'GD') #['GD', 'CG']

args   = parser.parse_args()

os.environ["CUDA_DEVICE_ORDER"]= "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]=str(args.gpu)
#seeds
torch.manual_seed(args.gen_seed)
np.random.seed(args.gen_seed)
device=torch.device('cuda')
batch_size=1
verbose = True

# load sample 
data_file = './test_data/test_dog_64x64.pt'
gt_img = torch.load(data_file)[None].cuda()

# forward model for specific inverse problem
H = gt_img.shape[-2]
W = gt_img.shape[-1]
if args.task=='deconv':
    # psf = gaussuian_filter(kernel_size=gt_img.shape[-1],sigma=0.05)
    # psf = torch.tensor(psf)[None,None].cuda()
    psf = torch.tensor(torch.load('./test_data/psf_64x64.pt'))[None,None].cuda()
    utils = LSI_utils(psf=psf,H=H,W=W)
    A_forw = utils.forward
    A_norm = utils.normal
    A_adj = utils.adjoint
    meas = A_forw(gt_img)
    meas = 0.001*torch.randn_like(meas) + meas
    x_adj = A_adj(meas)
elif args.task=='inpaint':
    m = H*W//args.R
    utils = InPaint_utils(batch=batch_size, m=m,H=H,W=W)
    A_forw = utils.forward
    A_norm = utils.normal
    A_adj = utils.adjoint
    meas = A_forw(gt_img)
    x_adj = A_adj(meas)
elif args.task=='compsens':
    m = H*W//args.R
    utils = CS_utils(batch=batch_size, m=m,H=H,W=W,device=device)
    A_forw = utils.forward
    A_norm = utils.normal
    A_adj = utils.adjoint
    meas = utils.meas_forward(gt_img)
    x_adj = A_adj(meas)

# cont = torch.load('/home/blevac/diffuser-cam/DiffuserCam-Tutorial/tutorial/diffsuer_data.pt')
# print(cont.keys())
# psf = torch.tensor(cont['psf'])[None,None].cuda()
# meas = torch.tensor(cont['meas'])[None,None].cuda()
# print(psf.shape)
# print(meas.shape)
# # forward model for specific inverse problem
# H = psf.shape[-2]
# W = psf.shape[-1]
# batch_size=1

# utils = LSI_utils(psf=psf,H=H,W=W)
# A_forw = utils.forward
# A_adj = utils.adjoint
# A_norm = utils.normal
# x_adj = A_adj(meas)


# designate + create save directory
results_dir = f'./results/{args.task}_results_likelihood/'

if not os.path.exists(results_dir):
    os.makedirs(results_dir)

x_cur = torch.zeros_like(x_adj)


if args.solver =='GD':
    recon_img = grad_descent(x=x_cur, meas=meas, A_forw=A_forw, max_iter=args.num_steps, l2lam = args.lam, step_sz=args.l_ss, proj = args.proj, device=device)
elif args.solver == 'CG':
    recon_img, _ = conjgrad(x=x_cur, b=x_adj, normal=A_norm, max_iter=args.num_steps, l2lam=args.lam, eps=1e-4, verbose=verbose, complex=True, device=device)


# gt_img=x_adj
img_nrmse = nrmse(gt_img, recon_img).item()
print('NRMSE: %.3f'%img_nrmse)

dict = { 
        'gt_img': gt_img,
        'recon':recon_img,
        'meas':meas,
        'forward_utils':utils,
        'nrmse':img_nrmse,
}

torch.save(dict, results_dir + '/checkpoint.pt')


