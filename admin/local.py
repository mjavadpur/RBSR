'''
Author: yishionsunshine 2267205780@qq.com
Date: 2022-11-10 13:14:16
LastEditors: yishionsunshine 2267205780@qq.com
LastEditTime: 2023-06-20 18:34:18
FilePath: /deep-burst-sr-master-L1-BasicVSRpp-shiyanshijiqun/admin/local.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
class EnvironmentSettings:
    def __init__(self):
        self.workspace_dir = './work_dirs'
        self.tensorboard_dir = self.workspace_dir + '/tensorboard/'    # Directory for tensorboard files.
        self.pretrained_nets_dir = './pretrained_networks'    # Directory for pre-trained networks.
        self.save_data_path = self.workspace_dir + '/evaluation'    # Directory for saving network predictions for evaluation.
        self.zurichraw2rgb_dir = '../dataset/Zurich-RAW-to-DSLR-Dataset'    # Zurich RAW 2 RGB path
        self.burstsr_dir = '../dataset/burstsr_dataset'    # BurstSR dataset path
        self.synburstval_dir = '../dataset/syn_burst_val'    # SyntheticBurst validation set path
