#!/usr/bin/python3
#coding=utf-8

import os
import sys
import argparse
import json


def _get_cli_value(flag):
    prefix = f'{flag}='
    for idx, arg in enumerate(sys.argv[1:], start=1):
        if arg == flag and idx + 1 < len(sys.argv):
            return sys.argv[idx + 1]
        if arg.startswith(prefix):
            return arg[len(prefix):]
    return None


gpu_override = _get_cli_value('--gpu')
if gpu_override is not None:
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu_override
else:
    os.environ.setdefault('CUDA_VISIBLE_DEVICES', '1')

import dataset

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from PIL import Image

from lscloss import *

import torchvision.utils as vutils
import numpy as np
import random

from models.model import get_model, get_sod_model
from models.loss import *
from utils.utils import *
from tools.general.io_utils import *
from tools.general.time_utils import *
from tools.general.json_utils import *
from tools.ai.log_utils import *
from tools.ai.demo_utils import *
from tools.ai.torch_utils import *
from tools.ai.evaluate_utils import *
from tools.ai.augment_utils import *
from tools.ai.randaugment import *

import infer

DEFAULT_SAM_BG_IOU_THRESH = 0.15
DEFAULT_SAM_LARGE_TARGET_HEAT_IOU_THRESH = 0.15
DEFAULT_SAM_LARGE_TARGET_BG_IOU_THRESH = 0.30
DEFAULT_SAM_AREA_LIMIT = 0.85
DEFAULT_SAM_LARGE_UNCERTAIN_AREA_THRESH = 0.30
DEFAULT_SAM_RULE_A_HEAT_IOU_THRESH = 0.40
DEFAULT_SAM_RULE_B_HEAT_IOU_DELTA = 0.05
DEFAULT_SAM_SMALL_FG_BOX_THRESH = 0.04
DEFAULT_DECODER_INIT_CHECKPOINT = './experiments/models/localizer_test-9.pth'
DEFAULT_DECODER_WARMUP_EPOCHS = 1
DEFAULT_DECODER_CCAM_LOSS_WEIGHT = 0.05
DEFAULT_DECODER_IOU_LOSS_WEIGHT = 1.0
DEFAULT_DECODER_STATIC_PSEUDO = False


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def count_parameters(model):
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return params / 1000000


def clip_gradient(optimizer, grad_clip):
    for group in optimizer.param_groups:
        for param in group['params']:
            if param.grad is not None:
                param.grad.data.clamp_(-grad_clip, grad_clip)


def create_directory(path):
    if not os.path.isdir(path):
        os.makedirs(path)
    return path


class PolyOptimizer(torch.optim.SGD):
    def __init__(self, params, lr, weight_decay, max_step, momentum=0.9):
        super().__init__(params, lr, weight_decay)
        self.global_step = 0
        self.max_step = max_step
        self.momentum = momentum
        self._initial_lr = [group['lr'] for group in self.param_groups]

    def step(self, closure=None):
        if self.global_step < self.max_step:
            lr_mult = (1 - self.global_step / self.max_step) ** self.momentum
            for idx, group in enumerate(self.param_groups):
                group['lr'] = self._initial_lr[idx] * lr_mult

        super().step(closure)
        self.global_step += 1


def build_sam_helper_preserve_rng(args, save_root):
    python_rng_state = random.getstate()
    numpy_rng_state = np.random.get_state()
    torch_rng_state = torch.get_rng_state()
    cuda_rng_state = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None

    try:
        from sam_helper import SAMTrainHelper

        helper = SAMTrainHelper(
            checkpoint=args.sam_checkpoint,
            save_root=save_root,
            model_type=args.sam_model_type,
            device=args.sam_device,
            area_limit=DEFAULT_SAM_AREA_LIMIT,
            score_thresh=args.sam_score_thresh,
            heat_iou_thresh=args.sam_heat_iou_thresh,
            large_target_heat_iou_thresh=DEFAULT_SAM_LARGE_TARGET_HEAT_IOU_THRESH,
            bg_iou_thresh=DEFAULT_SAM_BG_IOU_THRESH,
            large_target_bg_iou_thresh=DEFAULT_SAM_LARGE_TARGET_BG_IOU_THRESH,
            large_area_thresh=args.sam_large_area_thresh,
            large_uncertain_area_thresh=DEFAULT_SAM_LARGE_UNCERTAIN_AREA_THRESH,
            rule_a_heat_iou_thresh=DEFAULT_SAM_RULE_A_HEAT_IOU_THRESH,
            rule_b_heat_iou_delta=DEFAULT_SAM_RULE_B_HEAT_IOU_DELTA,
            resize_short_edge=args.sam_resize_short_edge,
            use_crf=args.sam_use_crf,
            small_fg_box_thresh=DEFAULT_SAM_SMALL_FG_BOX_THRESH,
        )
    finally:
        random.setstate(python_rng_state)
        np.random.set_state(numpy_rng_state)
        torch.set_rng_state(torch_rng_state)
        if cuda_rng_state is not None:
            torch.cuda.set_rng_state_all(cuda_rng_state)

    return helper


def denormalize_image(image):
    mean = torch.tensor([0.485, 0.456, 0.406], device=image.device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=image.device).view(1, 3, 1, 1)
    return (image * std + mean).clamp(0, 1)


def _strip_module_prefix(key):
    return key[7:] if key.startswith('module.') else key


def load_localizer_weights_into_decoder(model, checkpoint_path):
    if not checkpoint_path or not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f'Localizer checkpoint not found: {checkpoint_path}')

    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    source_state = checkpoint.get('state_dict', checkpoint) if isinstance(checkpoint, dict) else checkpoint
    target_state = model.state_dict()
    mapped_state = {}

    for raw_key, value in source_state.items():
        key = _strip_module_prefix(raw_key)
        candidates = [key]
        if key.startswith('backbone.'):
            candidates.append('encoder.' + key[len('backbone.'):])

        for candidate_key in candidates:
            if (
                candidate_key in target_state
                and candidate_key.startswith(('encoder.', 'ac_head.'))
                and target_state[candidate_key].shape == value.shape
            ):
                mapped_state[candidate_key] = value
                break

    if not mapped_state:
        raise RuntimeError(f'No encoder/ac_head weights matched from {checkpoint_path}')

    target_state.update(mapped_state)
    model.load_state_dict(target_state, strict=True)
    flag = bool(checkpoint.get('flag', False)) if isinstance(checkpoint, dict) else False
    return flag, len(mapped_state)


def _sample_meta_value(sample_meta, key, idx, default=0):
    if sample_meta is None or key not in sample_meta:
        return int(default)

    value = sample_meta[key]
    if isinstance(value, torch.Tensor):
        return int(value[idx].item())
    if isinstance(value, np.ndarray):
        return int(value[idx])
    if isinstance(value, (list, tuple)):
        return int(value[idx])
    return int(value)


def _pseudo_label_path(save_root, prefix, epoch, sample_name):
    stem = os.path.splitext(os.path.basename(str(sample_name)))[0]
    return os.path.join(save_root, prefix, f'epoch{epoch + 1}', f'{stem}.png')


def _print_sam_progress(epoch, processed, total):
    if total <= 0:
        message = f'\rSAM pseudo labels epoch {epoch + 1}: {processed}'
    else:
        ratio = min(max(float(processed) / float(total), 0.0), 1.0)
        bar_width = 30
        filled = int(round(bar_width * ratio))
        bar = '=' * filled + '.' * (bar_width - filled)
        message = f'\rSAM pseudo labels epoch {epoch + 1}: [{bar}] {processed}/{total} ({ratio * 100:.1f}%)'
    sys.stdout.write(message)
    sys.stdout.flush()


def _read_pseudo_label(path, target_hw, binary=False):
    target_h, target_w = target_hw
    resample = Image.NEAREST if binary else Image.BILINEAR
    with Image.open(path) as image:
        image = image.convert('L')
        if image.size != (target_w, target_h):
            image = image.resize((target_w, target_h), resample=resample)
        label = np.asarray(image, dtype=np.float32) / 255.0

    if binary:
        label = (label > 0.5).astype(np.float32)
    return label.astype(np.float32)


def load_pseudo_binary_batch(sample_names, sample_meta, save_root, pseudo_epoch, target_hw, device):
    binary_maps = []
    missing = []

    for sample_idx, sample_name in enumerate(sample_names):
        binary_path = _pseudo_label_path(save_root, 'pseudo_labels_binary', pseudo_epoch, sample_name)

        if not os.path.exists(binary_path):
            missing.append(binary_path)
            continue

        binary_map = _read_pseudo_label(binary_path, target_hw, binary=True)

        if _sample_meta_value(sample_meta, 'flipped', sample_idx, 0):
            binary_map = np.ascontiguousarray(binary_map[:, ::-1])

        binary_maps.append(binary_map)

    if missing:
        preview = ', '.join(missing[:3])
        raise FileNotFoundError(f'Missing binary pseudo labels for epoch {pseudo_epoch + 1}: {preview}')

    binary_tensor = torch.from_numpy(np.stack(binary_maps, axis=0)).unsqueeze(1).to(device=device, dtype=torch.float32)
    return binary_tensor


def save_hyperparams(save_root, tag, args, train_config, batch_size, epochs, use_decoder):
    payload = {
        'tag': tag,
        'train_mode': args.train_mode,
        'use_decoder': use_decoder,
        'batch_size': batch_size,
        'epochs': epochs,
        'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES', '0'),
        'optimizer_config': train_config,
        'sam_config': {
            'use_sam_final': args.use_sam_final,
            'sam_checkpoint': args.sam_checkpoint,
            'sam_model_type': args.sam_model_type,
            'sam_device': args.sam_device,
            'sam_score_thresh': args.sam_score_thresh,
            'sam_heat_iou_thresh': args.sam_heat_iou_thresh,
            'sam_large_target_heat_iou_thresh': DEFAULT_SAM_LARGE_TARGET_HEAT_IOU_THRESH,
            'sam_bg_iou_thresh': DEFAULT_SAM_BG_IOU_THRESH,
            'sam_large_target_bg_iou_thresh': DEFAULT_SAM_LARGE_TARGET_BG_IOU_THRESH,
            'sam_area_limit': DEFAULT_SAM_AREA_LIMIT,
            'sam_large_uncertain_area_thresh': DEFAULT_SAM_LARGE_UNCERTAIN_AREA_THRESH,
            'sam_rule_a_heat_iou_thresh': DEFAULT_SAM_RULE_A_HEAT_IOU_THRESH,
            'sam_rule_b_heat_iou_delta': DEFAULT_SAM_RULE_B_HEAT_IOU_DELTA,
            'sam_small_fg_box_thresh': DEFAULT_SAM_SMALL_FG_BOX_THRESH,
            'sam_large_area_thresh': args.sam_large_area_thresh,
            'sam_resize_short_edge': args.sam_resize_short_edge,
            'sam_use_crf': args.sam_use_crf,
        },
        'decoder_pseudo_config': {
            'decoder_init_checkpoint': args.decoder_init_checkpoint,
            'decoder_warmup_epochs': args.decoder_warmup_epochs,
            'decoder_iou_target': 'pseudo_labels_binary',
            'decoder_static_pseudo': args.decoder_static_pseudo,
            'decoder_skip_warmup': args.decoder_skip_warmup,
            'decoder_pseudo_root': args.decoder_pseudo_root or save_root,
            'decoder_pseudo_epoch': args.decoder_pseudo_epoch,
            'decoder_disable_ccam_loss': args.decoder_disable_ccam_loss,
            'decoder_ccam_loss_weight': args.decoder_ccam_loss_weight,
            'decoder_effective_ccam_loss_weight': 0.0 if args.decoder_disable_ccam_loss else args.decoder_ccam_loss_weight,
            'decoder_iou_loss_weight': args.decoder_iou_loss_weight,
        },
        'command': ' '.join(sys.argv),
        'argv': sys.argv,
    }

    out_path = os.path.join(save_root, 'hyperparams.json')
    with open(out_path, 'w', encoding='utf-8') as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def export_localizer_outputs(loader, model, flag, save_root, epoch, sam_helper=None):
    was_training = model.training
    model.eval()
    total_samples = len(loader.dataset) if hasattr(loader, 'dataset') else 0
    processed_samples = 0

    try:
        for image, sample_names, sample_meta in loader:
            image = image.type(torch.FloatTensor).cuda()
            gt_label, bg_label, cam_map = infer.light_cam(model, image, flag, return_cam=True)

            save_label_visuals(image, gt_label, sample_names, os.path.join(save_root, 'location-label'), epoch)
            save_fg_bg_overlay_visuals(image, gt_label, bg_label, sample_names, os.path.join(save_root, 'location-label'), epoch)

            if sam_helper is not None:
                sam_helper.process_batch(
                    denormalize_image(image),
                    cam_map,
                    gt_label,
                    bg_label,
                    list(sample_names),
                    sample_meta,
                    epoch,
                )
                processed_samples += len(sample_names)
                _print_sam_progress(epoch, processed_samples, total_samples)
        if sam_helper is not None:
            sys.stdout.write('\n')
            sys.stdout.flush()
    finally:
        if was_training:
            model.train()


def save_label_visuals(images, labels, sample_names, save_root, epoch):
    images = denormalize_image(images.detach())
    labels = (labels.detach() > 0.5).float()

    epoch_dir = os.path.join(save_root, f'epoch_{epoch:02d}')
    binary_dir = create_directory(os.path.join(epoch_dir, 'binary'))
    overlay_dir = create_directory(os.path.join(epoch_dir, 'overlay'))

    for sample_idx in range(images.size(0)):
        image_cpu = images[sample_idx].cpu()
        label_cpu = labels[sample_idx].cpu()
        sample_name = sample_names[sample_idx]

        if label_cpu.dim() == 2:
            label_cpu = label_cpu.unsqueeze(0)

        binary_mask = label_cpu.repeat(3, 1, 1)
        overlay = image_cpu.clone()
        color_mask = torch.zeros_like(overlay)
        color_mask[0] = 1.0
        mask_region = binary_mask[0] > 0.5
        overlay[:, mask_region] = 0.6 * overlay[:, mask_region] + 0.4 * color_mask[:, mask_region]

        vutils.save_image(binary_mask, os.path.join(binary_dir, f'{sample_name}.png'), padding=0)
        vutils.save_image(overlay, os.path.join(overlay_dir, f'{sample_name}.png'), padding=0)


def save_fg_bg_overlay_visuals(images, fg_labels, bg_labels, sample_names, save_root, epoch):
    images = denormalize_image(images.detach())
    fg_labels = (fg_labels.detach() > 0.5).float()
    bg_labels = (bg_labels.detach() > 0.5).float()

    epoch_dir = os.path.join(save_root, f'epoch_{epoch:02d}')
    overlay_dir = create_directory(os.path.join(epoch_dir, 'fg_bg_overlay'))

    for sample_idx in range(images.size(0)):
        image_cpu = images[sample_idx].cpu()
        fg_cpu = fg_labels[sample_idx].cpu()
        bg_cpu = bg_labels[sample_idx].cpu()
        sample_name = sample_names[sample_idx]

        if fg_cpu.dim() == 2:
            fg_cpu = fg_cpu.unsqueeze(0)
        if bg_cpu.dim() == 2:
            bg_cpu = bg_cpu.unsqueeze(0)

        overlay = image_cpu.clone()
        fg_region = fg_cpu[0] > 0.5
        bg_region = bg_cpu[0] > 0.5

        fg_color = torch.zeros_like(overlay)
        fg_color[0] = 1.0
        bg_color = torch.zeros_like(overlay)
        bg_color[2] = 1.0

        overlay[:, bg_region] = 0.6 * overlay[:, bg_region] + 0.4 * bg_color[:, bg_region]
        overlay[:, fg_region] = 0.6 * overlay[:, fg_region] + 0.4 * fg_color[:, fg_region]

        vutils.save_image(overlay, os.path.join(overlay_dir, f'{sample_name}.png'), padding=0)


def IOU(pred, target):
    inter = target * pred
    union = target + pred - target * pred
    iou_loss = 1 - torch.sum(inter, dim=(1, 2, 3)) / (torch.sum(union, dim=(1, 2, 3)) + 1e-7)
    return iou_loss.mean()


def IOU_Loss(preds, target):
    loss = 0

    target = target.gt(0.5).float()
    preds = nn.functional.interpolate(preds, size=target.size()[-2:], mode='bilinear')
    loss += IOU(torch.sigmoid(preds), target)

    return loss


loss_lsc_kernels_desc_defaults = [{"weight": 1, "xy": 6, "rgb": 0.1}]
loss_lsc_radius = 5


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train-mode', choices=['decoder', 'localizer'], default='decoder')
    parser.add_argument('--gpu', default=None, help='CUDA_VISIBLE_DEVICES value')
    parser.add_argument('--tag', default=None, help='Optional experiment tag override')
    parser.add_argument('--use-sam-final', action='store_true', help='Run SAM only in the final epoch for visualization.')
    parser.add_argument('--sam-checkpoint', default=None, help='SAM checkpoint path.')
    parser.add_argument('--sam-model-type', default='vit_h', help='SAM model type.')
    parser.add_argument('--sam-device', default='cuda', help='Device for SAM inference.')
    parser.add_argument('--sam-score-thresh', type=float, default=0.60, help='Minimum SAM predicted IoU score.')
    parser.add_argument('--sam-heat-iou-thresh', type=float, default=0.1, help='Minimum IoU with activation region.')
    parser.add_argument('--sam-large-area-thresh', type=float, default=0.06, help='Area threshold for using 5 prompts.')
    parser.add_argument('--sam-resize-short-edge', type=int, default=640, help='Resize short edge before SAM inference.')
    parser.add_argument('--sam-use-crf', action='store_true', help='Apply CRF refinement before saving pseudo labels.')
    parser.add_argument('--decoder-init-checkpoint', default=DEFAULT_DECODER_INIT_CHECKPOINT, help='Localizer checkpoint used to initialize encoder and ac_head in decoder mode.')
    parser.add_argument('--decoder-warmup-epochs', type=int, default=DEFAULT_DECODER_WARMUP_EPOCHS, help='Decoder-mode epochs used only for CCAM+SAM pseudo-label generation.')
    parser.add_argument('--decoder-ccam-loss-weight', type=float, default=DEFAULT_DECODER_CCAM_LOSS_WEIGHT, help='CCAM contrastive loss weight after decoder warmup.')
    parser.add_argument('--decoder-disable-ccam-loss', action='store_true', help='Disable CCAM contrastive loss in decoder mode.')
    parser.add_argument('--decoder-bce-loss-weight', type=float, default=0.0, help=argparse.SUPPRESS)
    parser.add_argument('--decoder-iou-loss-weight', type=float, default=DEFAULT_DECODER_IOU_LOSS_WEIGHT, help='IoU pseudo-label loss weight in decoder mode.')
    parser.add_argument('--decoder-static-pseudo', action='store_true', default=DEFAULT_DECODER_STATIC_PSEUDO, help='Use only warmup CCAM+SAM pseudo labels for all decoder training epochs.')
    parser.add_argument('--decoder-skip-warmup', action='store_true', help='Skip decoder warmup/SAM generation and train all epochs from existing pseudo labels.')
    parser.add_argument('--decoder-pseudo-root', default=None, help='Root that contains pseudo_labels_binary/epoch*/. Defaults to the current moco_see tag directory.')
    parser.add_argument('--decoder-pseudo-epoch', type=int, default=0, help='Zero-based pseudo-label epoch to load when using static or skip-warmup pseudo labels. 0 means epoch1.')
    return parser.parse_args()


def train(Dataset, args):
    default_tag = 'moco_v2_sod_seg' if args.train_mode == 'decoder' else 'moco_v2_localizer'
    TAG = args.tag or default_tag
    BATCH_SIZE = 8
    TRAIN_EPOCHS = 10
    alpha = 0.25
    use_decoder = args.train_mode == 'decoder'
    decoder_warmup_epochs = args.decoder_warmup_epochs if use_decoder else 0
    effective_warmup_epochs = 0 if (use_decoder and args.decoder_skip_warmup) else decoder_warmup_epochs
    EPOCHS = TRAIN_EPOCHS + effective_warmup_epochs if use_decoder else TRAIN_EPOCHS
    decoder_ccam_loss_weight = 0.0 if (use_decoder and args.decoder_disable_ccam_loss) else args.decoder_ccam_loss_weight
    if use_decoder and (not args.decoder_skip_warmup) and decoder_warmup_epochs < 1:
        raise ValueError('--decoder-warmup-epochs must be >= 1 for decoder mode unless --decoder-skip-warmup is set')
    if use_decoder and args.decoder_pseudo_epoch < 0:
        raise ValueError('--decoder-pseudo-epoch must be >= 0')

    log_dir = create_directory('./experiments/logs/')
    log_path = log_dir + '{}.txt'.format(TAG)
    model_dir = create_directory('./experiments/models/')
    model_path = model_dir + '{}.pth'.format(TAG)
    cam_path = './experiments/images/{}'.format(TAG)
    create_directory(cam_path)
    create_directory(cam_path + '/train')
    create_directory(cam_path + '/test')
    create_directory(cam_path + '/train/colormaps')
    create_directory(cam_path + '/test/colormaps')

    train_timer = Timer()

    cfg = Dataset.Config(datapath='./dataset/DUTS-TR/', savepath='./experiments/models/', mode='train', batch=BATCH_SIZE, lr=0.0025, momen=0.9, decay=5e-4, epoch=EPOCHS)
    data = Dataset.UData(cfg)
    loader = DataLoader(data, batch_size=cfg.batch, shuffle=True, pin_memory=True, num_workers=4)
    pseudo_loader = None
    if use_decoder:
        pseudo_cfg = Dataset.Config(
            datapath='./dataset/DUTS-TR/',
            savepath='./experiments/models/',
            mode='train',
            batch=BATCH_SIZE,
            lr=0.0025,
            momen=0.9,
            decay=5e-4,
            epoch=EPOCHS,
            disable_flip=True,
        )
        pseudo_data = Dataset.UData(pseudo_cfg)
        pseudo_loader = DataLoader(pseudo_data, batch_size=pseudo_cfg.batch, shuffle=False, pin_memory=True, num_workers=4)

    net = get_sod_model() if use_decoder else get_model(pretrained='mocov2')

    use_gpu = os.environ.get('CUDA_VISIBLE_DEVICES', '0')

    the_number_of_gpu = len(use_gpu.split(','))
    print(use_gpu)
    if the_number_of_gpu > 1:
        print('preparing data parallel')
        net = nn.DataParallel(net)
    net.train()
    net.cuda()
    model_for_optim = net.module if isinstance(net, nn.DataParallel) else net
    flag = False
    if use_decoder:
        flag, loaded_count = load_localizer_weights_into_decoder(model_for_optim, args.decoder_init_checkpoint)
        print(f'Loaded {loaded_count} encoder/ac_head tensors from {args.decoder_init_checkpoint}; flag={flag}')

    criterion = [SimMaxLoss(metric='cos', alpha=alpha).cuda(), SimMinLoss(metric='cos').cuda(),
                 SimMaxLoss(metric='cos', alpha=alpha).cuda()]

    if use_decoder:
        config = {
            'optim': 'SGD',
            'lr': 0.005,
            'epoch': EPOCHS,
            'step_size': [EPOCHS],
            'gamma': 0.1,
            'clip_gradient': 0,
            'train_epochs': TRAIN_EPOCHS,
            'total_epochs': EPOCHS,
            'warmup_epochs': effective_warmup_epochs,
            'skip_warmup': args.decoder_skip_warmup,
            'ccam_loss_enabled': not args.decoder_disable_ccam_loss,
            'ccam_loss_weight': decoder_ccam_loss_weight,
            'iou_loss_weight': args.decoder_iou_loss_weight,
        }
        encoder_module = model_for_optim.encoder
        module_lr = [
            {'params': encoder_module.parameters(), 'lr': config['lr'] / 10, 'weight_decay': 0.00005},
            {'params': model_for_optim.ac_head.parameters(), 'lr': config['lr'] / 10, 'weight_decay': 0.00005},
            {'params': model_for_optim.decoder.parameters(), 'lr': config['lr'], 'weight_decay': 0.0005, 'momentum': 0.9, 'nesterov': True},
        ]
        optimizer = torch.optim.SGD(params=module_lr)
        scheduler = None
    else:
        config = {
            'optim': 'PolyOptimizer+CosineAnnealingLR',
            'lr': 0.0001,
            'weight_decay': 0.0001,
            'epoch': EPOCHS,
            'clip_gradient': 0,
        }
        param_groups = model_for_optim.get_parameter_groups()
        max_step = max(1, len(loader) * EPOCHS)
        optimizer = PolyOptimizer([
            {'params': param_groups[0], 'lr': config['lr'], 'weight_decay': config['weight_decay']},
            {'params': param_groups[1], 'lr': 2 * config['lr'], 'weight_decay': 0},
            {'params': param_groups[2], 'lr': 10 * config['lr'], 'weight_decay': config['weight_decay']},
            {'params': param_groups[3], 'lr': 20 * config['lr'], 'weight_decay': 0},
        ], lr=config['lr'], weight_decay=config['weight_decay'], max_step=max_step)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, max_step)

    train_meter = Average_Meter(['loss', 'positive_loss', 'negative_loss', 'iou_loss', 'lsc_loss', 'reg_loss'])
    data_dic = {
        'train': [],
        'validation': []
    }
    log_func = lambda string='': log_print(string, log_path)

    tmp_path = create_directory(os.path.join('./moco_see', TAG))
    pseudo_label_root = args.decoder_pseudo_root or tmp_path
    sam_final_helper = None
    needs_decoder_sam = use_decoder and not args.decoder_skip_warmup
    if (needs_decoder_sam or args.use_sam_final) and not args.sam_checkpoint:
        raise ValueError('--sam-checkpoint is required for decoder pseudo-label training or --use-sam-final')
    if needs_decoder_sam:
        sam_final_helper = build_sam_helper_preserve_rng(args, tmp_path)
    if use_decoder and args.decoder_skip_warmup:
        log_func(f'[i]\tSkipping decoder warmup/SAM generation; loading pseudo labels from {pseudo_label_root}/pseudo_labels_binary/epoch{args.decoder_pseudo_epoch + 1}')

    save_hyperparams(tmp_path, TAG, args, config, BATCH_SIZE, EPOCHS, use_decoder)

    torch.cuda.empty_cache()

    for epoch in range(EPOCHS):

        if use_decoder and epoch < effective_warmup_epochs:
            log_func(f'[i]\tEpoch[{epoch:,}/{EPOCHS:,}], warmup: generating CCAM+SAM pseudo labels')
            export_localizer_outputs(
                pseudo_loader,
                net,
                flag,
                tmp_path,
                epoch,
                sam_helper=sam_final_helper,
            )
            sam_final_helper.finalize_epoch(epoch)
        else:
            for i, (image, sample_names, sample_meta) in enumerate(loader):
                image = image.type(torch.FloatTensor).cuda()

                optimizer.zero_grad()
                fg_feats, bg_feats, ccam, decoder_out = net(image)

                if (not use_decoder) and epoch == 0 and i == (len(loader) - 1):
                    flag = check_positive(ccam)
                    print(f"Is Negative: {flag}")
                if flag:
                    ccam = 1 - ccam

                if use_decoder and args.decoder_disable_ccam_loss:
                    loss1 = torch.tensor(0.0, device=image.device)
                    loss2 = torch.tensor(0.0, device=image.device)
                    loss3 = torch.tensor(0.0, device=image.device)
                    ccam_loss = torch.tensor(0.0, device=image.device)
                else:
                    loss1 = criterion[0](fg_feats)
                    loss2 = criterion[1](bg_feats, fg_feats)
                    loss3 = criterion[2](bg_feats)
                    ccam_loss = loss1 + loss2 + loss3

                if use_decoder:
                    if args.decoder_skip_warmup or args.decoder_static_pseudo:
                        pseudo_epoch = args.decoder_pseudo_epoch
                    else:
                        pseudo_epoch = epoch - effective_warmup_epochs
                    pseudo_binary = load_pseudo_binary_batch(
                        sample_names,
                        sample_meta,
                        pseudo_label_root,
                        pseudo_epoch,
                        decoder_out.shape[-2:],
                        image.device,
                    )
                    iou_loss = IOU_Loss(decoder_out, pseudo_binary)
                    loss2_lsc = torch.tensor(0.0, device=image.device)
                    reg_loss = torch.tensor(0.0, device=image.device)
                    seg_loss = args.decoder_iou_loss_weight * iou_loss
                    loss = decoder_ccam_loss_weight * ccam_loss + seg_loss
                else:
                    iou_loss = torch.tensor(0.0, device=image.device)
                    loss2_lsc = torch.tensor(0.0, device=image.device)
                    reg_loss = torch.tensor(0.0, device=image.device)
                    loss = ccam_loss

                loss.backward()
                optimizer.step()
                if scheduler is not None:
                    scheduler.step()

                train_meter.add({
                    'loss': loss.item(),
                    'positive_loss': loss1.item() + loss3.item(),
                    'negative_loss': loss2.item(),
                    'iou_loss': iou_loss.item(),
                    'lsc_loss': loss2_lsc.item(),
                    'reg_loss': reg_loss.item(),
                })

                if i % 20 == 0:
                    visualize_heatmap(TAG, image.clone().detach(), ccam, 0, i)
                    loss, positive_loss, negative_loss, iou_loss_value, lsc_loss, reg_loss = train_meter.get(clear=True)
                    learning_rate = float(get_learning_rate_from_optimizer(optimizer))

                    data = {
                        'epoch': epoch,
                        'max_epoch': EPOCHS,
                        'iteration': i + 1,
                        'learning_rate': learning_rate,
                        'loss': loss,
                        'positive_loss': positive_loss,
                        'negative_loss': negative_loss,
                        'iou_loss': iou_loss_value,
                        'lsc_loss': lsc_loss,
                        'reg_loss': reg_loss,
                        'time': train_timer.tok(clear=True),
                    }
                    data_dic['train'].append(data)

                    log_func('[i]\t'
                             'Epoch[{epoch:,}/{max_epoch:,}],\t'
                             'iteration={iteration:,}, \t'
                             'learning_rate={learning_rate:.8f}, \t'
                             'loss={loss:.4f}, \t'
                             'positive_loss={positive_loss:.4f}, \t'
                             'negative_loss={negative_loss:.4f}, \t'
                             'iou_loss={iou_loss:.4f}, \t'
                             'lsc_loss={lsc_loss:.4f}, \t'
                             'reg_loss={reg_loss:.4f}, \t'
                             'time={time:.0f}sec'.format(**data)
                             )

            if use_decoder and (not args.decoder_skip_warmup) and (not args.decoder_static_pseudo) and epoch < EPOCHS - 1:
                log_func(f'[i]\tEpoch[{epoch:,}/{EPOCHS:,}], refreshing CCAM+SAM pseudo labels for next epoch')
                export_localizer_outputs(
                    pseudo_loader,
                    net,
                    flag,
                    tmp_path,
                    epoch,
                    sam_helper=sam_final_helper,
                )
                sam_final_helper.finalize_epoch(epoch)

        if not use_decoder and epoch == EPOCHS - 1:
            if args.use_sam_final:
                if sam_final_helper is None:
                    sam_final_helper = build_sam_helper_preserve_rng(args, tmp_path)
            export_localizer_outputs(
                loader,
                net,
                flag,
                tmp_path,
                epoch,
                sam_helper=sam_final_helper,
            )
            if sam_final_helper is not None:
                sam_final_helper.finalize_epoch(epoch)

        torch.save({
            'state_dict': net.module.state_dict() if (the_number_of_gpu > 1) else net.state_dict(),
            'flag': flag,
            'mode': args.train_mode,
        }, cfg.savepath + f'{TAG}-{epoch}.pth')


if __name__ == '__main__':
    args = parse_args()
    set_seed(7)
    train(dataset, args)
