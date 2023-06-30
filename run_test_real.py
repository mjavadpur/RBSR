
import imp
import os
import sys

env_path = os.path.join(os.path.dirname(__file__), '../..')
if env_path not in sys.path:
    sys.path.append(env_path)

import torch
import numpy as np
import tqdm
import numpy as np
import tqdm
from models.alignment.pwcnet import PWCNet
import random
from dataset.burstsr_dataset import get_burstsr_val_set, CanonImage
import cv2

def setup_seed(seed=0):
	torch.manual_seed(seed)
	torch.cuda.manual_seed(seed)
	torch.cuda.manual_seed_all(seed)
	np.random.seed(seed)
	random.seed(seed)
	# torch.backends.cudnn.enabled = True
	torch.backends.cudnn.deterministic = True
	torch.backends.cudnn.benchmark = True

def compute_score_BIPnet(model, model_path):
    from utils.metrics import AlignedPSNR, AlignedLPIPS, AlignedSSIM
    device = 'cuda'             
    checkpoint_dict = torch.load(model_path, map_location='cpu')
    model.load_state_dict(checkpoint_dict['net'])
    model = model.to(device).train(False)
    model.eval()
    model.cuda() 

    alignment_net = PWCNet(load_pretrained=True, weights_path='pretrained_networks/pwcnet-network-default.pth')
    alignment_net = alignment_net.cuda()
    alignment_net = alignment_net.eval()
    aligned_psnr_fn = AlignedPSNR(alignment_net=alignment_net)
    aligned_ssim_fn = AlignedSSIM(alignment_net=alignment_net)

    dataset = get_burstsr_val_set()
    PSNR = []
    LPIPS = []
    SSIM = []

    for idx in tqdm.tqdm(range(len(dataset))):
        data = dataset[idx]
        gt = data['frame_gt'].unsqueeze(0)
        burst = data['burst'].unsqueeze(0)
        burst_name = data['burst_name']

        burst = burst.to(device)
        gt = gt.to(device)

        with torch.no_grad():
            burst = burst[:, :14, ...]
            net_pred, _ = model(burst)
            output = net_pred.clamp(0.0, 1.0)

        PSNR_temp = aligned_psnr_fn(output, gt, burst).cpu().numpy()            
        PSNR.append(PSNR_temp)
        
        SSIM_temp = aligned_ssim_fn(output, gt, burst).cpu().numpy()
        SSIM.append(SSIM_temp)
    print(f'PSNR: {np.mean(np.array(PSNR))}, SSIM: {np.mean(np.array(SSIM))}')
    return np.mean(np.array(PSNR)), np.mean(np.array(SSIM))
if __name__ == "__main__":
    setup_seed(0)
    from models.RBSR_test import RBSRNet
    net = RBSRNet()
    path = "./pretrained_networks/RBSR_realwporld.pth.tar"
    psnr, ssim = compute_score_BIPnet(net, path)

