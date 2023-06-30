'''
Author: yishionsunshine 2267205780@qq.com
Date: 2023-06-20 17:56:15
LastEditors: yishionsunshine 2267205780@qq.com
LastEditTime: 2023-06-20 21:17:06
FilePath: /deep-burst-sr-master-L1-BasicVSRpp-shiyanshijiqun/simple_valid.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''

# Copyright (c) 2021 Huawei Technologies Co., Ltd.
# Licensed under CC BY-NC-SA 4.0 (Attribution-NonCommercial-ShareAlike 4.0 International) (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode
#
# The code is released for academic research use only. For commercial use, please contact Huawei Technologies Co., Ltd.
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import imp
import os
import sys

env_path = os.path.join(os.path.dirname(__file__), '../..')
if env_path not in sys.path:
    sys.path.append(env_path)

from dataset.synthetic_burst_val_set import SyntheticBurstVal
import torch

from models.loss.image_quality_v2 import PSNR, SSIM
import numpy as np
import tqdm
import random
def setup_seed(seed=0):
	torch.manual_seed(seed)
	torch.cuda.manual_seed(seed)
	torch.cuda.manual_seed_all(seed)
	np.random.seed(seed)
	random.seed(seed)
	# torch.backends.cudnn.enabled = True
	torch.backends.cudnn.deterministic = True
	torch.backends.cudnn.benchmark = True


def compute_score(model, model_path=""):
    device = 'cuda'
    net = model
    if model_path is not None:
        checkpoint_dict = torch.load(model_path, map_location='cpu')
        net.load_state_dict(checkpoint_dict['net'])
    net = net.to(device).train(False)
    
    dataset = SyntheticBurstVal()
    metrics = ('psnr', 'ssim')
    boundary_ignore = 40
    metrics_all = {}
    scores = {}
    for m in metrics:
        if m == 'psnr':
            loss_fn = PSNR(boundary_ignore=boundary_ignore)
        elif m == 'ssim':
            loss_fn = SSIM(boundary_ignore=boundary_ignore, use_for_loss=False)
        else:
            raise Exception
        metrics_all[m] = loss_fn
        scores[m] = []

    scores_all = {}
    scores = {k: [] for k, v in scores.items()}
    for idx in tqdm.tqdm(range(len(dataset))):
        burst, gt, meta_info = dataset[idx]
        burst_name = meta_info['burst_name']

        burst = burst.to(device).unsqueeze(0)
        gt = gt.to(device)

        with torch.no_grad():
            burst = burst[:, :14,...]
            net_pred, _ = net(burst)

        # Perform quantization to be consistent with evaluating on saved images
        net_pred_int = (net_pred.clamp(0.0, 1.0) * 2 ** 14).short()
        net_pred = net_pred_int.float() / (2 ** 14)

        for m, m_fn in metrics_all.items():
            metric_value = m_fn(net_pred, gt.unsqueeze(0)).cpu().item()
            scores[m].append(metric_value) 
    psnr_mean = np.mean(scores['psnr'])
    ssim_mean = np.mean(scores['ssim'])

    print(f'PSNR: {psnr_mean}, SSIM: {ssim_mean}')



if __name__ == "__main__":
    from models.RBSR_test import RBSRNet
    net = RBSRNet()
    total = sum([param.nelement() for param in net.parameters()])
    print("Number of parameter: %.2fM" % (total/1e6))
    setup_seed(seed=0)
    compute_score(net, "./pretrained_networks/RBSR_synthetic.pth.tar")