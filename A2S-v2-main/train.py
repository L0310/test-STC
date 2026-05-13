import sys
import os
import time
import random
import cv2
import importlib
import shutil
import numpy as np
from math import exp

from tqdm import tqdm
from collections import OrderedDict
from util import *
from PIL import Image
from data import Train_Dataset, Test_Dataset, get_loader, get_test_list
from test import test_model
import torch
import torch.nn.functional as F
from torch.nn import utils
from torch.utils.data import DataLoader
from base.framework_factory import load_framework

from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist

torch.set_printoptions(precision=5)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

DEFAULT_SAM_BG_IOU_THRESH = 0.15
DEFAULT_SAM_LARGE_TARGET_HEAT_IOU_THRESH = 0.15
DEFAULT_SAM_LARGE_TARGET_BG_IOU_THRESH = 0.30
DEFAULT_SAM_AREA_LIMIT = 0.85
DEFAULT_SAM_LARGE_UNCERTAIN_AREA_THRESH = 0.30
DEFAULT_SAM_RULE_A_HEAT_IOU_THRESH = 0.40
DEFAULT_SAM_RULE_B_HEAT_IOU_DELTA = 0.05
DEFAULT_SAM_SMALL_FG_BOX_THRESH = 0.04
DEFAULT_SAM_PSEUDO_SSD_PARENT = '/tmp/xiao_ssd_data/STC-main/A2S-v2-main/pseudo'
SAM_COMPACT_SAVE_PREFIXES = [
    'affinity_split_overlay',
    'affinity_superpixel_overlay',
    'mask_point_candidates',
    'mask_point_candidates_neg',
    'mask_point_sam_seg',
    'mask_point_sam_seg_neg',
    'point_candidates',
    'point_candidates_neg',
    'points_sam_seg',
    'points_sam_seg_neg',
    'pseudo_labels_binary',
    'pseudo_labels_neg_binary',
    'pseudo_labels_mask_point_binary',
    'pseudo_labels_mask_point_neg_binary',
    'pseudo_labels_rule_ab_binary',
    'pseudo_labels_neg_rule_ab_binary',
    'pseudo_labels_mask_point_rule_ab_binary',
    'pseudo_labels_mask_point_neg_rule_ab_binary',
    'sam_failures',
]
CCAM_SCALES = [1.0, 0.5, 1.5]


def use_cornet_sam_pseudo(net_name, config):
    return net_name == 'cornet' and config['stage'] > 1 and bool(config.get('use_sam_pseudo', False))


def use_cornet_mask_point_outputs(config):
    if config.get('sam_affinity_use_mask_prompt', False):
        return True
    prefixes = config.get('sam_save_prefixes', None)
    if not prefixes:
        return False
    return any(
        str(prefix).startswith('mask_point_')
        or str(prefix).startswith('pseudo_labels_mask_point')
        or str(prefix).startswith('sam_mask_point_')
        for prefix in prefixes
    )


def prepare_cornet_sam_config(config):
    if not config.get('sam_checkpoint'):
        raise ValueError('--sam-checkpoint is required when using cornet SAM pseudo-label training.')
    sam_root = config.get('sam_pseudo_root') or './pseudo/cornet_sam'
    sam_root = os.path.abspath(sam_root)
    sam_work_root = resolve_sam_pseudo_work_root(config, sam_root)
    config['sam_pseudo_final_root'] = sam_root
    config['sam_pseudo_work_root'] = sam_work_root
    config['sam_save_prefixes'] = list(SAM_COMPACT_SAVE_PREFIXES)
    config['pseudo_root'] = os.path.join(sam_root, 'pseudo_labels_binary', 'epoch1')
    config['allow_missing_gt'] = True
    if sam_work_root != sam_root:
        print('SAM pseudo labels will be staged under {} and synced to {}.'.format(sam_work_root, sam_root))
    return sam_work_root


def resolve_sam_pseudo_work_root(config, final_root):
    work_root = str(config.get('sam_pseudo_work_root', '') or '').strip()
    if work_root:
        return os.path.abspath(work_root)

    ssd_parent = str(config.get('sam_pseudo_ssd_parent', '') or DEFAULT_SAM_PSEUDO_SSD_PARENT).strip()
    if not ssd_parent:
        return os.path.abspath(final_root)

    a2s_pseudo_root = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'pseudo'))
    final_root = os.path.abspath(final_root)
    try:
        common = os.path.commonpath([a2s_pseudo_root, final_root])
    except ValueError:
        common = ''
    if common == a2s_pseudo_root:
        relative_root = os.path.relpath(final_root, a2s_pseudo_root)
    else:
        relative_root = os.path.basename(final_root)
    return os.path.abspath(os.path.join(ssd_parent, relative_root))


def reset_sam_pseudo_work_root(config):
    work_root = config.get('sam_pseudo_work_root')
    final_root = config.get('sam_pseudo_final_root')
    if not work_root or os.path.abspath(work_root) == os.path.abspath(final_root):
        return
    if os.path.exists(work_root):
        print('Removing stale SAM pseudo staging root: {}'.format(work_root))
        shutil.rmtree(work_root)
    os.makedirs(work_root, exist_ok=True)


def sync_and_cleanup_sam_pseudo(config):
    work_root = config.get('sam_pseudo_work_root')
    final_root = config.get('sam_pseudo_final_root')
    if not work_root or not final_root:
        return
    work_root = os.path.abspath(work_root)
    final_root = os.path.abspath(final_root)
    if work_root == final_root:
        return

    os.makedirs(final_root, exist_ok=True)
    for prefix in config.get('sam_save_prefixes', SAM_COMPACT_SAVE_PREFIXES):
        src = os.path.join(work_root, prefix)
        if not os.path.exists(src):
            continue
        dst = os.path.join(final_root, prefix)
        if os.path.exists(dst):
            shutil.rmtree(dst)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copytree(src, dst)
    print('Synced SAM pseudo labels from {} to {}.'.format(work_root, final_root))
    shutil.rmtree(work_root)
    print('Removed SAM pseudo staging root: {}'.format(work_root))


def resolve_sam_dino_weight(config):
    dino_weight = str(config.get('sam_dino_weight', '') or '').strip()
    if dino_weight:
        return dino_weight
    default_weight = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'PretrainModel', 'dinov2_vitl14_pretrain.pth')
    return default_weight if os.path.exists(default_weight) else ''


def resolve_sam_prompt_mode(config):
    if config.get('sam_disable_affinity_split', False):
        return 'points'
    mode = str(config.get('sam_prompt_mode', 'affinity') or 'affinity').strip().lower()
    if mode not in ('affinity', 'points'):
        raise ValueError('--sam-prompt-mode should be affinity or points, got {}'.format(mode))
    return mode


def build_sam_helper(config, save_root):
    prompt_mode = resolve_sam_prompt_mode(config)
    if prompt_mode == 'points':
        from sam_helper_bf import SAMTrainHelper

        if config.get('sam_affinity_use_mask_prompt', False):
            print('Using legacy SAM point prompts plus positive mask prompts from sam_helper_bf.py.')
        else:
            print('Using legacy SAM point prompts from sam_helper_bf.py.')
        return SAMTrainHelper(
            checkpoint=config['sam_checkpoint'],
            save_root=save_root,
            model_type=config.get('sam_model_type', 'vit_l'),
            device=config.get('sam_device', 'cuda'),
            area_limit=DEFAULT_SAM_AREA_LIMIT,
            score_thresh=config.get('sam_score_thresh', 0.60),
            heat_iou_thresh=config.get('sam_heat_iou_thresh', 0.10),
            large_target_heat_iou_thresh=DEFAULT_SAM_LARGE_TARGET_HEAT_IOU_THRESH,
            bg_iou_thresh=config.get('sam_bg_iou_thresh', DEFAULT_SAM_BG_IOU_THRESH),
            large_target_bg_iou_thresh=config.get('sam_large_target_bg_iou_thresh', DEFAULT_SAM_LARGE_TARGET_BG_IOU_THRESH),
            large_area_thresh=config.get('sam_large_area_thresh', 0.06),
            large_uncertain_area_thresh=DEFAULT_SAM_LARGE_UNCERTAIN_AREA_THRESH,
            rule_a_heat_iou_thresh=DEFAULT_SAM_RULE_A_HEAT_IOU_THRESH,
            rule_b_heat_iou_delta=DEFAULT_SAM_RULE_B_HEAT_IOU_DELTA,
            resize_short_edge=config.get('sam_resize_short_edge', 640),
            use_crf=config.get('sam_use_crf', False),
            small_fg_box_thresh=DEFAULT_SAM_SMALL_FG_BOX_THRESH,
            use_mask_prompt=config.get('sam_affinity_use_mask_prompt', False),
        )

    from sam_helper import SAMTrainHelper

    mask_point_outputs = use_cornet_mask_point_outputs(config)
    if mask_point_outputs:
        print('Using affinity SAM point prompts plus selected mask+point outputs from sam_helper.py.')
    else:
        print('Using affinity SAM point prompts from sam_helper.py.')

    return SAMTrainHelper(
        checkpoint=config['sam_checkpoint'],
        save_root=save_root,
        model_type=config.get('sam_model_type', 'vit_l'),
        device=config.get('sam_device', 'cuda'),
        area_limit=DEFAULT_SAM_AREA_LIMIT,
        score_thresh=config.get('sam_score_thresh', 0.60),
        heat_iou_thresh=config.get('sam_heat_iou_thresh', 0.10),
        large_target_heat_iou_thresh=DEFAULT_SAM_LARGE_TARGET_HEAT_IOU_THRESH,
        bg_iou_thresh=config.get('sam_bg_iou_thresh', DEFAULT_SAM_BG_IOU_THRESH),
        large_target_bg_iou_thresh=config.get('sam_large_target_bg_iou_thresh', DEFAULT_SAM_LARGE_TARGET_BG_IOU_THRESH),
        large_area_thresh=config.get('sam_large_area_thresh', 0.06),
        large_uncertain_area_thresh=DEFAULT_SAM_LARGE_UNCERTAIN_AREA_THRESH,
        resize_short_edge=config.get('sam_resize_short_edge', 640),
        use_crf=config.get('sam_use_crf', False),
        small_fg_box_thresh=DEFAULT_SAM_SMALL_FG_BOX_THRESH,
        use_affinity_split=True,
        depth_root=config.get('sam_depth_root', ''),
        seed_points_per_instance=config.get('sam_seed_points_per_instance', 3),
        affinity_use_mask_prompt=mask_point_outputs,
        use_negative_prompt=not config.get('sam_disable_neg_prompt', False),
        neg_ccam_thresh=config.get('sam_neg_ccam_thresh', 0.25),
        neg_bg_thresh=config.get('sam_neg_bg_thresh', 0.05),
        neg_box_expand=config.get('sam_neg_box_expand', 0.15),
        neg_margin=config.get('sam_neg_margin', 8),
        neg_points_per_component=config.get('sam_neg_points_per_component', 3),
        save_prefixes=config.get('sam_save_prefixes', None),
        dino_weight=resolve_sam_dino_weight(config),
        dino_model=config.get('sam_dino_model', 'dinov2_vitl14'),
        dino_repo=config.get('sam_dino_repo', ''),
        dino_device=config.get('sam_dino_device', '') or config.get('sam_device', 'cuda'),
        dino_max_side=config.get('sam_dino_max_side', 700),
        dino_pca_dim=config.get('sam_dino_pca_dim', 64),
    )


def denormalize_image(image):
    mean_t = torch.tensor([0.485, 0.456, 0.406], device=image.device).view(1, 3, 1, 1)
    std_t = torch.tensor([0.229, 0.224, 0.225], device=image.device).view(1, 3, 1, 1)
    return (image * std_t + mean_t).clamp(0, 1)


def ccam_flag_from_batch(ccam):
    edge_mean = (
        ccam[:, 0, 0:4, :].mean()
        + ccam[:, 0, :, 0:4].mean()
        + ccam[:, 0, -4:, :].mean()
        + ccam[:, 0, :, -4:].mean()
    ) / 4
    return bool((edge_mean > 0.5).detach().cpu().item())


def forward_cornet_cam(model, inputs):
    return model(inputs, 'cam')


def infer_cornet_ccam(model, inputs, flag, hith=0.55, loth=0.15):
    cam_list = []
    b, _, h, w = inputs.shape
    with torch.no_grad():
        for scale in CCAM_SCALES:
            if scale == 1.0:
                scaled_inputs = inputs
            else:
                scaled_inputs = F.interpolate(inputs, size=(int(scale * h), int(scale * w)), mode='bilinear', align_corners=False)

            inputs_cat = torch.cat([scaled_inputs, scaled_inputs.flip(-1)], dim=0)
            raw_cam = forward_cornet_cam(model, inputs_cat)
            if flag:
                raw_cam = -raw_cam
            raw_cam = F.interpolate(raw_cam, size=(h, w), mode='bilinear', align_corners=False)
            raw_cam = torch.max(raw_cam[:b], raw_cam[b:].flip(-1))
            cam_list.append(F.relu(raw_cam))

        cam = torch.sum(torch.stack(cam_list, dim=0), dim=0)
        cam = cam + F.adaptive_max_pool2d(-cam, (1, 1))
        cam = cam / (F.adaptive_max_pool2d(cam, (1, 1)) + 1e-5)

        fg = torch.zeros_like(cam)
        fg[cam >= hith] = 1.0
        bg = torch.zeros_like(cam)
        bg[cam <= loth] = 1.0

    return fg, bg, cam


def make_cornet_sam_loader(config):
    sam_config = dict(config)
    sam_config['disable_flip'] = True
    sam_config['allow_missing_gt'] = True
    dataset = Train_Dataset(sam_config)
    return DataLoader(
        dataset=dataset,
        batch_size=config['batch'],
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        drop_last=False,
    )


def set_stage_lr(optim, config, stage, epoch, mul, pack=None):
    if stage == 1:
        tune = 2 if pack is not None and 'MSB-TR' in pack['name'][0] else 20
        if epoch > tune:
            optim.param_groups[0]['lr'] = config['lr'] * mul * 0.01
        else:
            optim.param_groups[0]['lr'] = 0
        optim.param_groups[1]['lr'] = config['lr'] * mul
        return

    if len(optim.param_groups) >= 3 and hasattr(optim, 'param_groups'):
        optim.param_groups[0]['lr'] = config['lr'] * mul * 0.1
        optim.param_groups[1]['lr'] = config['lr'] * mul * 0.1
        optim.param_groups[2]['lr'] = config['lr'] * mul
    else:
        optim.param_groups[0]['lr'] = config['lr'] * mul * 0.1
        optim.param_groups[1]['lr'] = config['lr'] * mul


def set_cornet_warmup_lr(optim, config):
    if len(optim.param_groups) >= 3:
        optim.param_groups[0]['lr'] = config['lr'] / 10
        optim.param_groups[1]['lr'] = config['lr'] / 10
        optim.param_groups[2]['lr'] = config['lr']


def build_cornet_warmup_optimizer(model, config):
    target = model.module if hasattr(model, 'module') else model
    return torch.optim.SGD(
        [
            {'params': target.encoder.parameters(), 'lr': config['lr'] / 10, 'weight_decay': 0.00005},
            {'params': target.ac_head.parameters(), 'lr': config['lr'] / 10, 'weight_decay': 0.00005},
            {
                'params': target.decoder.parameters(),
                'lr': config['lr'],
                'weight_decay': 0.0005,
                'momentum': 0.9,
                'nesterov': True,
            },
        ]
    )


def set_model_layer4_stride(model, stride):
    target = model.module if hasattr(model, 'module') else model
    target.set_layer4_stride(stride)


def run_cornet_ccam_warmup(model, train_loader, optim, config, ccam_loss_func):
    model.train()
    num_iter = len(train_loader)
    batch_idx = 0
    loss_count, pos_count, neg_count = 0, 0, 0
    last_ccam = None
    optim.zero_grad()
    bar = tqdm(total=num_iter, desc='cornet-sam | warmup:')
    st = time.time()

    for i, pack in enumerate(train_loader, start=1):
        set_cornet_warmup_lr(optim, config)

        images = pack['image'].float().cuda()
        preds = model(images, 'ccam_train')
        loss1, loss2, loss3, loss = ccam_loss_func(preds)
        loss.backward()
        last_ccam = preds['ccam'].detach()

        batch_idx += 1
        if batch_idx == config['ave_batch']:
            if config['clip_gradient']:
                utils.clip_grad_norm_(model.parameters(), config['clip_gradient'])
            optim.step()
            optim.zero_grad()
            batch_idx = 0

        loss_count += loss.data
        pos_count += loss1.data + loss3.data
        neg_count += loss2.data
        lrs = ','.join([format(param['lr'], ".2e") for param in optim.param_groups])
        bar.set_postfix_str(
            '{:4}/{:4} | loss: {:1.3f}, pos: {:1.3f}, neg: {:1.3f}, LRs: [{}], time: {:1.3f}.'.format(
                i, num_iter, float(loss_count / i), float(pos_count / i), float(neg_count / i), lrs, time.time() - st
            ),
            refresh=False,
        )
        bar.update(1)

    if batch_idx > 0:
        if config['clip_gradient']:
            utils.clip_grad_norm_(model.parameters(), config['clip_gradient'])
        optim.step()
        optim.zero_grad()

    bar.close()
    if last_ccam is None:
        return False
    flag = ccam_flag_from_batch(last_ccam)
    print('cornet-sam warmup CCAM polarity flag: {}.'.format(flag))
    return flag


def export_cornet_sam_pseudo(model, loader, config, sam_helper, flag):
    was_training = model.training
    model.eval()
    total = len(loader.dataset)
    processed = 0
    hith = config.get('ccam_hith', 0.55)
    loth = config.get('ccam_loth', 0.15)

    with torch.no_grad():
        for pack in tqdm(loader, desc='cornet-sam | SAM pseudo'):
            images = pack['image'].float().cuda()
            fg, bg, cam = infer_cornet_ccam(model, images, flag, hith=hith, loth=loth)
            sample_names = list(pack.get('image_name', pack['name']))
            sample_meta = pack.get('meta', None)
            sam_helper.process_batch(
                denormalize_image(images),
                cam,
                fg,
                bg,
                sample_names,
                sample_meta,
                0,
            )
            processed += len(sample_names)
    sam_helper.finalize_epoch(0)
    print('Generated SAM pseudo labels for {} samples.'.format(processed if processed else total))
    if was_training:
        model.train()


def main():
    # Loading model
    if len(sys.argv) > 1:
        net_name = sys.argv[1]
    else:
        print('Need model name!')
        return
        
    # Loading model
    config, model, optim, sche, model_loss, saver = load_framework(net_name)
    config['net_name'] = net_name
    stage = config['stage']
    cornet_sam_enabled = use_cornet_sam_pseudo(net_name, config)
    sam_pseudo_save_root = None
    if cornet_sam_enabled:
        sam_pseudo_save_root = prepare_cornet_sam_config(config)
    
    if config['weight'] != '':
        print('Load weights from: {}.'.format(config['weight']))
        model.load_state_dict(torch.load(config['weight'], map_location='cpu'))
        
    train_loader = get_loader(config)
    if cornet_sam_enabled:
        ccam_loss_func = getattr(importlib.import_module('methods.{}.loss'.format(net_name)), 'ccam_loss')
        warmup_optim = build_cornet_warmup_optimizer(model, config)
        flag = run_cornet_ccam_warmup(model, train_loader, warmup_optim, config, ccam_loss_func)
        del warmup_optim
        sam_loader = make_cornet_sam_loader(config)
        reset_sam_pseudo_work_root(config)
        sam_helper = build_sam_helper(config, sam_pseudo_save_root)
        export_cornet_sam_pseudo(model, sam_loader, config, sam_helper, flag)
        del sam_helper
        sync_and_cleanup_sam_pseudo(config)
        if config.get('decoder_train_layer4_stride') is not None:
            set_model_layer4_stride(model, config['decoder_train_layer4_stride'])
            print('Switched cornet layer4 stride to {} for decoder training.'.format(config['decoder_train_layer4_stride']))
        torch.cuda.empty_cache()
    
    test_sets = get_test_list(config['vals'], config)
    
    debug = config['debug']
    num_epoch = config['epoch']
    num_iter = len(train_loader)
    ave_batch = config['ave_batch']
    #batch = ave_batch * config['batch']
    trset = config['trset']
    batch_idx = 0
    model.zero_grad()
    for epoch in range(1, num_epoch + 1):
        model.train()
        torch.cuda.empty_cache()
        
        if debug:
            test_model(model, test_sets, config, epoch)
        
        bar = tqdm(
            total=num_iter,
            desc='{:10}-{:8} | epoch {:2}:'.format(net_name, config['sub'], epoch),
        )
        
        st = time.time()
        loss_count, adb_count, ac_count, mse_count = 0, 0, 0, 0
        optim.zero_grad()
        
        fin_lr = 0.2
        for i, pack in enumerate(train_loader, start=1):
            cur_it = i + (epoch-1) * num_iter
            total_it = num_epoch * num_iter
            itr = (1 - cur_it / total_it) * (1 - fin_lr) + fin_lr
            mul = itr
            
            set_stage_lr(optim, config, stage, epoch, mul, pack)
                
            images = pack['image'].float()
            gts = pack['gt'].float()
            gt_names = pack['name']
            flips = pack['flip']
            
            images, gts = images.cuda(), gts.cuda()
            
            priors = [images]
            if 'dep' in pack.keys():
                priors.append(pack['dep'].float().cuda())
            if 'of' in pack.keys():
                priors.append(pack['of'].float().cuda())
            if 'th' in pack.keys():
                priors.append(pack['th'].float().cuda())
            
            loss = 0
            if stage == 1:
                Y = model(images, 'train')
                
                config['param'] = tran_param(config)
                images_temp = transform(images, False, config)
                priors = torch.cat(priors, dim=1)
                priors_temp = transform(priors, False, config)
                
                Y_ref = model(images_temp, 'train')
                
                lr_weight = np.array(config['lrw'].split(',')).astype(np.float)
                if lr_weight is None or len(lr_weight) != 3:
                    lr_weight = [0.5, 0.05, 1]
                
                loss0, loss1, loss2 = model_loss(Y, priors, Y_ref, priors_temp, epoch, lr_weight, config, gt_names)
                loss += loss0 + loss1 + loss2
                    
                ac_count += loss1
                mse_count += loss2
                
            elif stage > 1:
                Y = model(priors, 'train')
                loss0 = model_loss(Y, gts.gt(0.5).float(), config)
                loss = loss0
                
            loss_count += loss.data
            adb_count += loss0
            loss.backward()

            batch_idx += 1
            if batch_idx == ave_batch:
                if config['clip_gradient']:
                    utils.clip_grad_norm_(model.parameters(), config['clip_gradient'])
                optim.step()
                optim.zero_grad()
                batch_idx = 0
            
            lrs = ','.join([format(param['lr'], ".2e") for param in optim.param_groups])
            if stage == 1:
                bar.set_postfix_str(
                    '{:4}/{:4} | loss: {:1.3f}, dfs: {:1.3f}, bac: {:1.3f}, mse: {:1.3f}, LRs: [{}], time: {:1.3f}.'.format(
                        i, num_iter, float(loss_count / i), float(adb_count / i), float(ac_count / i), float(mse_count / i), lrs, time.time() - st
                    ),
                    refresh=False,
                )
            else:
                bar.set_postfix_str(
                    '{:4}/{:4} | loss: {:1.3f}, LRs: [{}], time: {:1.3f}.'.format(
                        i, num_iter, float(loss_count / i), lrs, time.time() - st
                    ),
                    refresh=False,
                )
            bar.update(1)
            
            if epoch > 1 and stage > 1 and config['olr']:
                lamda = config.get('resdual', 0.4)
                for gt_path, pred, gt, flip in zip(gt_names, torch.sigmoid(Y['final'].detach()), gts, flips):
                    pred = F.interpolate(pred.unsqueeze(0), size=gt.size()[1:], mode='bilinear', align_corners=True)[0]
                    if flip:
                        pred = pred.flip(2)
                        gt = gt.flip(2)

                    new_gt = pred * (1 - lamda) + gt * lamda
                    new_gt = new_gt / (new_gt.max() + 1e-8)
                    new_gt = (new_gt.squeeze(0).cpu().numpy() * 255).astype(np.uint8)
                    cv2.imwrite(gt_path, new_gt)
                
        sche.step()
        bar.close()
        if stage == 1:
            print('| loss: {:1.3f}, dfs: {:1.3f}, bac: {:1.3f}, mse: {:1.3f}, LRs: [{}], time: {:1.3f}.'.format(
                float(loss_count / num_iter), float(adb_count / num_iter), float(ac_count / num_iter), float(mse_count / num_iter), lrs, time.time() - st))
        else:
            print('| loss: {:1.3f}, LRs: [{}], time: {:1.3f}.'.format(
                float(loss_count / num_iter), lrs, time.time() - st))
        torch.cuda.empty_cache()
        test_model(model, test_sets, config, epoch)

if __name__ == "__main__":
    main()
