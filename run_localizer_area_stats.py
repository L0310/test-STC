#!/usr/bin/python3
# coding=utf-8

import argparse
import glob
import json
import os
import subprocess
import sys


def parse_args():
    parser = argparse.ArgumentParser(
        description='Run localizer training and summarize activation uncertain-area ratios.'
    )
    parser.add_argument('--hith', type=float, default=0.6, help='Foreground threshold passed to infer.light_cam via env HITH.')
    parser.add_argument('--loth', type=float, default=0.1, help='Background threshold passed to infer.light_cam via env LOTH.')
    parser.add_argument('--gpu', default='1', help='CUDA_VISIBLE_DEVICES value used for both training and analysis.')
    parser.add_argument('--tag', default='exp_localizer_CCAMlr_e20', help='Experiment tag passed to train.py.')
    parser.add_argument('--train-script', default='./train.py', help='Training entry script.')
    parser.add_argument('--train-mode', default='localizer', choices=['localizer', 'decoder'], help='Training mode.')
    parser.add_argument('--datapath', default='./dataset/DUTS-TR/', help='Dataset path used for post-training analysis.')
    parser.add_argument('--num-workers', type=int, default=4, help='Dataloader workers for analysis.')
    parser.add_argument('--output-json', default=None, help='Optional explicit output json path.')
    parser.add_argument('--checkpoint', default=None, help='Optional explicit checkpoint path. If omitted, use the latest checkpoint for the tag.')
    parser.add_argument('--skip-train', action='store_true', help='Skip training and only analyze an existing checkpoint.')
    return parser.parse_args()


def run_training(args):
    env = os.environ.copy()
    env['HITH'] = str(args.hith)
    env['LOTH'] = str(args.loth)

    cmd = [
        sys.executable,
        args.train_script,
        '--train-mode',
        args.train_mode,
        '--gpu',
        args.gpu,
        '--tag',
        args.tag,
    ]

    print('running:', ' '.join(cmd))
    subprocess.run(cmd, check=True, env=env)


def resolve_checkpoint(tag, explicit_checkpoint=None):
    if explicit_checkpoint:
        if not os.path.exists(explicit_checkpoint):
            raise FileNotFoundError(f'Checkpoint not found: {explicit_checkpoint}')
        return explicit_checkpoint

    pattern = os.path.join('./experiments/models', f'{tag}-*.pth')
    checkpoints = glob.glob(pattern)
    if not checkpoints:
        raise FileNotFoundError(f'No checkpoints found for tag={tag} under ./experiments/models/')

    def _epoch_key(path):
        stem = os.path.splitext(os.path.basename(path))[0]
        try:
            return int(stem.rsplit('-', 1)[-1])
        except ValueError:
            return -1

    checkpoints.sort(key=_epoch_key)
    return checkpoints[-1]


def analyze_checkpoint(args, checkpoint_path):
    os.environ['HITH'] = str(args.hith)
    os.environ['LOTH'] = str(args.loth)
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    import torch
    from torch.utils.data import DataLoader

    import dataset
    import infer
    from models.model import get_model

    cfg = dataset.Config(
        datapath=args.datapath,
        savepath='./experiments/models/',
        mode='test',
        batch=1,
        lr=0.0,
        momen=0.0,
        decay=0.0,
        epoch=1,
    )
    data = dataset.UData(cfg)
    loader = DataLoader(
        data,
        batch_size=1,
        shuffle=False,
        pin_memory=True,
        num_workers=args.num_workers,
    )

    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model = get_model(pretrained='mocov2')
    model.load_state_dict(checkpoint['state_dict'], strict=True)
    model = model.cuda()
    model.eval()

    flag = bool(checkpoint.get('flag', False))
    records = []

    with torch.no_grad():
        for image, _, sample_names in loader:
            image = image.float().cuda(non_blocking=True)
            fg_label, bg_label, cam_map = infer.light_cam(model, image, flag, return_cam=True)

            fg_ratio = float(fg_label.float().mean().item())
            bg_ratio = float(bg_label.float().mean().item())
            uncertain_with_fg_ratio = float(1.0 - bg_ratio)
            uncertain_only_ratio = float(((cam_map > args.loth) & (cam_map < args.hith)).float().mean().item())

            records.append({
                'name': sample_names[0],
                'uncertain_with_fg_area_ratio': uncertain_with_fg_ratio,
                'uncertain_only_area_ratio': uncertain_only_ratio,
                'fg_area_ratio': fg_ratio,
                'bg_area_ratio': bg_ratio,
            })

    small_target_large_uncertain_images = [
        item for item in records
        if item['fg_area_ratio'] < 0.05 and item['uncertain_with_fg_area_ratio'] > 0.4
    ]

    summary = {
        'total_images': len(records),
        'small_target_count': sum(item['fg_area_ratio'] < 0.05 for item in records),
        'fg_area_lt_0_05_count': sum(item['fg_area_ratio'] < 0.05 for item in records),
        'small_target_uncertain_with_fg_gt_0_4_count': len(small_target_large_uncertain_images),
        'area_gt_0_5_count': sum(item['uncertain_with_fg_area_ratio'] > 0.5 for item in records),
        'area_gt_0_4_count': sum(item['uncertain_with_fg_area_ratio'] > 0.4 for item in records),
        'area_gt_0_3_count': sum(item['uncertain_with_fg_area_ratio'] > 0.3 for item in records),
        'area_lt_0_3_count': sum(item['uncertain_with_fg_area_ratio'] < 0.3 for item in records),
        'small_target_definition': 'small target = fg_area_ratio < 0.05',
        'small_target_uncertain_with_fg_gt_0_4_definition': 'fg_area_ratio < 0.05 and uncertain_with_fg_area_ratio > 0.4',
        'definition': 'uncertain_with_fg_area_ratio = area(cam > LOTH) = 1 - bg_area_ratio',
    }

    return {
        'summary': summary,
        'settings': {
            'tag': args.tag,
            'train_mode': args.train_mode,
            'checkpoint': checkpoint_path,
            'datapath': args.datapath,
            'hith': args.hith,
            'loth': args.loth,
            'gpu': args.gpu,
        },
        'images': records,
        'small_target_uncertain_with_fg_gt_0_4_images': sorted(
            small_target_large_uncertain_images,
            key=lambda item: item['uncertain_with_fg_area_ratio'],
            reverse=True,
        ),
    }


def build_sorted_payload(payload, images, sort_definition):
    return {
        'summary': payload['summary'],
        'settings': payload['settings'],
        'sort_definition': sort_definition,
        'images': images,
    }


def main():
    args = parse_args()

    if not args.skip_train:
        run_training(args)

    checkpoint_path = resolve_checkpoint(args.tag, args.checkpoint)
    payload = analyze_checkpoint(args, checkpoint_path)

    default_output = os.path.join('./moco_see', args.tag, 'uncertain_area_stats.json')
    output_path = args.output_json or default_output
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)

    output_root, output_ext = os.path.splitext(output_path)

    small_target_images = sorted(
        [
            item for item in payload['images']
            if item['fg_area_ratio'] < 0.05
        ],
        key=lambda item: item['fg_area_ratio'],
    )
    small_target_fg_sorted_path = f'{output_root}_small_target_fg_sorted{output_ext}'
    with open(small_target_fg_sorted_path, 'w', encoding='utf-8') as handle:
        json.dump(
            build_sorted_payload(
                payload,
                small_target_images,
                'small targets only (fg_area_ratio < 0.05), sorted by fg_area_ratio ascending',
            ),
            handle,
            ensure_ascii=False,
            indent=2,
        )

    uncertain_with_fg_sorted_images = sorted(
        payload['images'],
        key=lambda item: item['uncertain_with_fg_area_ratio'],
    )
    uncertain_with_fg_sorted_path = f'{output_root}_uncertain_with_fg_sorted{output_ext}'
    with open(uncertain_with_fg_sorted_path, 'w', encoding='utf-8') as handle:
        json.dump(
            build_sorted_payload(
                payload,
                uncertain_with_fg_sorted_images,
                'all images sorted by uncertain_with_fg_area_ratio ascending',
            ),
            handle,
            ensure_ascii=False,
            indent=2,
        )

    print(f'checkpoint={checkpoint_path}')
    print(f'saved_json={output_path}')
    print(f'saved_small_target_fg_sorted_json={small_target_fg_sorted_path}')
    print(f'saved_uncertain_with_fg_sorted_json={uncertain_with_fg_sorted_path}')
    print(json.dumps(payload['summary'], ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
