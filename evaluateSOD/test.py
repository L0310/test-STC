import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "2"
import torch
import torch.nn.functional as F
import sys
import numpy as np
import argparse
import cv2
import main as mm
import time
import torchvision.transforms as transforms
torch.cuda.set_device(0)


parser = argparse.ArgumentParser()
parser.add_argument('--testsize', type=int, default=224, help='testing size')
parser.add_argument('--gpu_id', type=str, default='0', help='select gpu id')
# parser.add_argument('--test_path',type=str,default='/storage/CJQ2/RGBD/dataset/replace/',help='test dataset path')
# parser.add_argument('--test_path',type=str,default='/storage/mobile/HRZ/segment-anything-main/input/PASCAL-S/',help='test dataset path')
opt = parser.parse_args()

# dataset_path = opt.test_path

sal_path = "/home/phd/Code/cjw/CJW/evaluateSOD/MDSAM/"
# test
# test_datasets = ['NLPR', 'NJU2K',   'DUT', 'STERE',  'LFSD','SIP']
# test_datasets = ['DUTS', 'DUT-O', 'ECSSD', 'HKU-IS', 'PASCAL-S']
test_datasets = ['ECSSD']
for dataset in test_datasets:
    save_path = os.path.join(sal_path,dataset)
    if not os.path.exists(save_path):
        continue
    # gt_root = '/storage/mobile/HRZ/DATA/VT821/GT/'
    # gt_root = '/storage/mobile/HRZ/DATA/VT1000/GT/'
    gt_root = '/home/phd/Code/cjw/CJW/DATA/ECSSD/test/mask/'
    
    _ = mm.evalateSOD(save_path, gt_root, dataset,'best')
    # end=time.clock()
    # print(str(end-start))
    print('Test Done!')
