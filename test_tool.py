#!/usr/bin/python3
#coding=utf-8

import os
from pathlib import Path
# os.environ['CUDA_VISIBLE_DEVICES'] = "0"

import sys
sys.path.insert(0, '../')
SALOD_ROOT = Path(__file__).resolve().parent / 'SALOD-master'
if str(SALOD_ROOT) not in sys.path:
    sys.path.insert(0, str(SALOD_ROOT))
sys.dont_write_bytecode = True
import copy

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import dataset as dataset_module
from torch.utils.data import DataLoader
from utils.utils import *
from models.model import get_model
from tools.ai.log_utils import *
from tools.ai.demo_utils import *
from tools.ai.optim_utils import *
from tools.ai.torch_utils import *
from tools.ai.evaluate_utils import *
from tools.ai.augment_utils import *
from tools.ai.randaugment import *

from base.metric import MetricRecorder, normalize_pil

from models.model import get_sod_model
import cmapy

def create_directory(path):
    if not os.path.isdir(path):
        os.makedirs(path)
    return path

def get_strided_size(orig_size, stride):
    return ((orig_size[0]-1)//stride+1, (orig_size[1]-1)//stride+1)

def check_positive(am):
    assert am.shape[0] == am.shape[1]
    n = am.shape[0]
    edge_mean = (am[0, :].mean() + am[n - 1, :].mean() + am[:, 0].mean() + am[:, n - 1].mean()) / 4
    return edge_mean > 0.5


IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def _image_files_by_stem(root):
    return {
        path.stem: path
        for path in Path(root).rglob("*")
        if path.is_file() and path.suffix.lower() in IMG_EXTS
    }


def evaluate_with_salod_metrics(pred_dir, gt_dir, dataset_name):
    pred_root = Path(pred_dir)
    gt_root = Path(gt_dir)
    pred_paths = sorted([p for p in pred_root.rglob("*") if p.is_file() and p.suffix.lower() in IMG_EXTS])
    if not pred_paths:
        raise RuntimeError(f"No prediction files found under {pred_root}")
    gt_by_stem = _image_files_by_stem(gt_root)

    matched_pairs = []
    for pred_path in pred_paths:
        gt_path = gt_by_stem.get(pred_path.stem)
        if gt_path is not None:
            matched_pairs.append((pred_path, gt_path))

    if not matched_pairs:
        raise FileNotFoundError(
            'no matched prediction files found for {}'.format(dataset_name)
        )

    recorder = MetricRecorder(len(matched_pairs))
    for pred_path, gt_path in tqdm(matched_pairs, desc=f"Evaluate {dataset_name}"):
        gt = np.array(Image.open(gt_path).convert('L'))
        pred_img = Image.open(pred_path).convert('L')
        pred = np.array(pred_img.resize(gt.shape[::-1]))

        pred, gt = normalize_pil(pred, gt)
        recorder.update(pre=pred, gt=gt)

    mae, (_, mean_f, *_), s_measure, e_measure, _ = recorder.show(bit_num=3)
    return {
        'MAE': mae,
        'Mean-F': mean_f,
        'Mean-E': e_measure,
        'S-measure': s_measure,
    }


DATASET_ALIASES = {
    'DUTS-TE': ['DUTS-TE', 'DUTS_test'],
    'DUT-OMRON': ['DUT-OMRON', 'DUT_O'],
    'ECSSD': ['ECSSD'],
    'HKU-IS': ['HKU-IS', 'HKU_IS'],
    'PASCAL-S': ['PASCAL-S', 'PASCAL_S'],
    'SOD': ['SOD'],
    'DUTS-TR': ['DUTS-TR'],
}


def resolve_dataset_path(dataset_path):
    dataset_name = Path(dataset_path).name
    search_roots = [Path('./dataset'), Path('./dataset/GT')]

    for root in search_roots:
        candidate = root / dataset_name
        if candidate.is_dir():
            return str(candidate)

    for alias in DATASET_ALIASES.get(dataset_name, [dataset_name]):
        for root in search_roots:
            candidate = root / alias
            if candidate.is_dir():
                return str(candidate)

    raise FileNotFoundError('cannot find dataset directory for {}'.format(dataset_path))


all_dataset = [
    './dataset/DUTS-TE',
    './dataset/ECSSD',
    './dataset/HKU-IS',
    './dataset/PASCAL-S',
    './dataset/DUT-OMRON',
    # './dataset/SOD',
    # './dataset/DUTS-TR',
]

duts_test = ['./dataset/DUTS-TE']


def test(epoch, only_duts_test=True, for_stage2=False):
    dataset = dataset_module
    assert epoch >= 0

    localtime = time.asctime( time.localtime(time.time()) )

    TAG = "moco_v2_sod_seg"
    experiment_name = TAG
    experiment_name += '@val'
    
    pred_dir = create_directory(f'./experiments/predictions/{experiment_name}/')
    if for_stage2:
        model_path = './out_2nd/' + f'{TAG}-{epoch}.pth'
    else:
        model_path = './experiments/models/' + f'{TAG}-{epoch}.pth'

    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std = [0.229, 0.224, 0.225]

    normalize_fn = Normalize(imagenet_mean, imagenet_std)

    scales = [float(scale) for scale in '0.5,1.0,1.5,2.0'.split(',')]

    model = get_sod_model()
    model = model.cuda()
    model.eval()

    ckpt = torch.load(model_path)
    flag = ckpt['flag']
    model.load_state_dict(ckpt['state_dict'])

    model.eval()

    pp = 'DUTS-TR'
    cam_path = create_directory(f'./vis_cam/{experiment_name}/{pp}')
    print(cam_path)

    if only_duts_test:
        record = f'moco_v2_duts_test'
    else: 
        record = f'moco_v2_all_test'

    if for_stage2:
        record = '2nd_' + record
        test_save_path = 'eval'
    else: 
        test_save_path = 'moco_res_out'
    logfile = record + '.txt' # 每测试完一个数据集记录一次

    with open(logfile, 'a') as f:
        f.write("\n------------cut off line--------------\n")
        f.write(str(localtime) + '\n')
        f.write(f'start testing epoch {epoch}\n')

    def test_single(Dataset, Path):
        Path = resolve_dataset_path(Path)
        print(Path)
        
        ## dataset
        cfg    = Dataset.Config(datapath=Path, mode='test') 
        test_dataset = Dataset.UData(cfg)
        loader = DataLoader(test_dataset, batch_size=12, shuffle=False, num_workers=8)

        with torch.no_grad():
            for image, (H, W), name in tqdm(loader):
                image = image.cuda().float()
                _, _, _, out = model(image)

                out = torch.sigmoid(out).cpu().numpy() * 255
                for i in range(out.shape[0]):
                    pred = cv2.resize(out[i, 0], dsize=(int(W[i]),int(H[i])), interpolation=cv2.INTER_LINEAR)
                    head = f'./{test_save_path}/maps/' + cfg.datapath.split('/')[-1]   
                    if not os.path.exists(head):
                        os.makedirs(head)
                    pred_u8 = np.clip(np.round(pred), 0, 255).astype(np.uint8)
                    cv2.imwrite(head + '/' + name[i] + '.png', pred_u8)
        
        if for_stage2:
            method = 'detector'
        else: 
            method = 'usod'
        res = {}
        gt_dir = dataset_module.resolve_mask_dir(Path)
        datasetname = Path.split('/')[-1]
        sm_dir = f'./{test_save_path}/maps/' + datasetname  # 'SM/'
        if not os.path.exists(sm_dir):
            res[datasetname] = {'MAE': 0, 'Mean-F': 0, 'Mean-E': 0, 'S-measure': 0}
            raise ValueError('sm_dir not exist.')

        print('Evaluate ' + method + ' ' + datasetname + '------')

        res[datasetname] = evaluate_with_salod_metrics(sm_dir, gt_dir, datasetname)

        with open(logfile, 'a') as f:  # 'a' 打开文件接着写
            f.write('{} {} get {:.3f} mae, {:.3f} s-measure, '
                    '{:.3f} mean-e, {:.3f} mean-f \n'.format(
                datasetname, method, res[datasetname]['MAE'], res[datasetname]['S-measure'],
                res[datasetname]['Mean-E'], res[datasetname]['Mean-F']))

        return res[datasetname]['MAE'], res[datasetname]['S-measure'], \
               res[datasetname]['Mean-E'], res[datasetname]['Mean-F']

    test_list = duts_test if only_duts_test else all_dataset
    ret = None
    for path in test_list:
        try:
            ret = test_single(dataset, path)
        except FileNotFoundError as error:
            print('Skip {}: {}'.format(path, error))

    if ret is None:
        raise FileNotFoundError('no available test dataset found in {}'.format(test_list))

    return list(ret)
        
if __name__ == '__main__':
    test(7, only_duts_test=False)
