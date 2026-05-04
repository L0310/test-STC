#!/usr/bin/python3
#coding=utf-8

import argparse
import os
import random
from pathlib import Path

from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.transforms import functional as TF

from models.model import get_sod_model


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def create_directory(path):
    if not os.path.isdir(path):
        os.makedirs(path)
    return path


def iou_loss(logits, target):
    target = target.gt(0.5).float()
    pred = torch.sigmoid(logits)

    inter = target * pred
    union = target + pred - target * pred
    loss = 1 - torch.sum(inter, dim=(1, 2, 3)) / (torch.sum(union, dim=(1, 2, 3)) + 1e-7)
    return loss.mean()


class PseudoMaskDataset(Dataset):
    def __init__(self, image_datapath, mask_datapath, list_path, image_dir='image', mask_dir='mask'):
        self.image_datapath = Path(image_datapath)
        self.mask_datapath = Path(mask_datapath)
        self.image_root = self._resolve_dir(self.image_datapath, image_dir, ['image', 'images'])
        self.mask_root = self._resolve_dir(
            self.mask_datapath,
            mask_dir,
            ['mask', 'masks', 'pseudo', 'pseudos', 'segmentation', 'segmentations', 'label', 'labels'],
        )
        self.list_path = Path(list_path)

        if not self.list_path.exists():
            raise FileNotFoundError('train list not found: {}'.format(self.list_path))

        with open(self.list_path, 'r') as lines:
            self.samples = [line.strip() for line in lines if line.strip()]

        self.mean = (0.485, 0.456, 0.406)
        self.std = (0.229, 0.224, 0.225)
        self.image_exts = ['.jpg', '.jpeg', '.png', '.bmp']
        self.mask_exts = ['.png', '.jpg', '.jpeg', '.bmp']

        print('dataset getting {} samples'.format(len(self.samples)))

    def _resolve_dir(self, root_path, requested_name, candidates):
        if requested_name in ['', '.', './']:
            return root_path

        requested_path = root_path / requested_name
        if requested_path.exists():
            return requested_path

        for candidate in candidates:
            candidate_path = root_path / candidate
            if candidate_path.exists():
                return candidate_path

        if any(path.is_file() for path in root_path.iterdir()):
            return root_path

        raise FileNotFoundError(
            'directory not found under {}. tried: {}'.format(
                root_path,
                ', '.join([requested_name] + [name for name in candidates if name != requested_name]),
            )
        )

    def _resolve_file(self, root, sample, exts):
        sample_path = root / sample
        candidates = [sample_path]
        stem = Path(sample).stem

        for ext in exts:
            candidates.append(root / (stem + ext))
            candidates.append(root / (stem + ext.upper()))

        for candidate in candidates:
            if candidate.exists():
                return candidate

        matches = sorted(root.glob(stem + '.*'))
        if matches:
            return matches[0]

        raise FileNotFoundError('failed to find file for {} under {}'.format(sample, root))

    def __getitem__(self, idx):
        sample = self.samples[idx]
        image_path = self._resolve_file(self.image_root, sample, self.image_exts)
        mask_path = self._resolve_file(self.mask_root, sample, self.mask_exts)

        image = Image.open(image_path).convert('RGB')
        mask = Image.open(mask_path).convert('L')

        image = TF.resize(image, [512, 512], interpolation=Image.BILINEAR)
        mask = TF.resize(mask, [512, 512], interpolation=Image.NEAREST)

        if random.random() < 0.5:
            image = TF.hflip(image)
            mask = TF.hflip(mask)

        top, left, height, width = transforms.RandomCrop.get_params(image, output_size=(448, 448))
        image = TF.crop(image, top, left, height, width)
        mask = TF.crop(mask, top, left, height, width)

        image = TF.resize(image, [352, 352], interpolation=Image.BILINEAR)
        mask = TF.resize(mask, [352, 352], interpolation=Image.NEAREST)

        image = TF.to_tensor(image)
        image = TF.normalize(image, mean=self.mean, std=self.std)

        mask = TF.to_tensor(mask)
        mask = mask.gt(0.5).float()

        return image, mask

    def __len__(self):
        return len(self.samples)


def load_checkpoint(model, checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    state_dict = checkpoint['state_dict'] if isinstance(checkpoint, dict) and 'state_dict' in checkpoint else checkpoint
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)

    if missing_keys:
        print('missing keys: {}'.format(len(missing_keys)))
    if unexpected_keys:
        print('unexpected keys: {}'.format(len(unexpected_keys)))


def parse_args():
    parser = argparse.ArgumentParser(description='Train encoder and decoder with pseudo labels using IoU loss only.')
    parser.add_argument('--datapath', default='./dataset/DUTS-TR/', help='shared dataset root when images and pseudo labels are under one directory')
    parser.add_argument('--image-root', default='', help='dataset root that contains training images')
    parser.add_argument('--mask-root', default='', help='dataset root that contains pseudo labels')
    parser.add_argument('--list-path', default='', help='optional explicit path to the training sample list')
    parser.add_argument('--list-name', default='train.txt', help='training sample list file name')
    parser.add_argument('--image-dir', default='image', help='image directory name under datapath')
    parser.add_argument('--mask-dir', default='mask', help='pseudo label directory name under datapath')
    parser.add_argument('--savepath', default='./experiments/models/', help='checkpoint directory')
    parser.add_argument('--tag', default='pseudo_iou', help='checkpoint prefix')
    parser.add_argument('--load', default='', help='optional checkpoint to load before training')
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--lr', type=float, default=0.005)
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--seed', type=int, default=7)
    return parser.parse_args()


def train():
    args = parse_args()
    set_seed(args.seed)

    save_dir = create_directory(args.savepath)
    image_root = args.image_root or args.datapath
    mask_root = args.mask_root or args.datapath
    list_candidates = [
        Path(args.list_path) if args.list_path else None,
        Path(image_root) / args.list_name,
        Path(mask_root) / args.list_name,
    ]
    list_path = None
    for candidate in list_candidates:
        if candidate and candidate.exists():
            list_path = candidate
            break

    if list_path is None:
        raise FileNotFoundError(
            'train list not found. tried: {}'.format(
                ', '.join(str(candidate) for candidate in list_candidates if candidate is not None)
            )
        )

    dataset = PseudoMaskDataset(
        image_root,
        mask_root,
        list_path,
        image_dir=args.image_dir,
        mask_dir=args.mask_dir,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=args.num_workers,
    )

    net = get_sod_model()
    if args.load:
        load_checkpoint(net, args.load)

    for param in net.ac_head.parameters():
        param.requires_grad = False

    try:
        use_gpu = os.environ['CUDA_VISIBLE_DEVICES']
    except KeyError:
        print('cuda visible device exception')
        use_gpu = '0'

    the_number_of_gpu = len(use_gpu.split(','))
    print(use_gpu)
    if the_number_of_gpu > 1:
        print('preparing data parallel')
        net = nn.DataParallel(net)

    net = net.cuda()
    net.train()

    model_for_optim = net.module if isinstance(net, nn.DataParallel) else net
    optimizer = torch.optim.SGD(
        [
            {'params': model_for_optim.encoder.parameters(), 'lr': args.lr / 10, 'weight_decay': 0.00005},
            {'params': model_for_optim.decoder.parameters(), 'lr': args.lr, 'weight_decay': 0.0005},
        ],
        momentum=0.9,
        nesterov=True,
    )

    for epoch in range(args.epochs):
        epoch_loss = 0.0

        for iteration, (image, mask) in enumerate(loader, start=1):
            image = image.cuda(non_blocking=True).float()
            mask = mask.cuda(non_blocking=True).float()

            optimizer.zero_grad()
            _, _, _, out_final = net(image)
            loss = iou_loss(out_final, mask)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

            if iteration % 20 == 0 or iteration == len(loader):
                print(
                    'Epoch[{}/{}] Iter[{}/{}] lr={:.8f} iou_loss={:.4f}'.format(
                        epoch + 1,
                        args.epochs,
                        iteration,
                        len(loader),
                        optimizer.param_groups[0]['lr'],
                        loss.item(),
                    )
                )

        mean_loss = epoch_loss / max(len(loader), 1)
        state_dict = net.module.state_dict() if isinstance(net, nn.DataParallel) else net.state_dict()
        save_file = os.path.join(save_dir, '{}-{}.pth'.format(args.tag, epoch))
        torch.save({'state_dict': state_dict, 'flag': False}, save_file)
        print('Epoch[{}/{}] mean_iou_loss={:.4f} saved={}'.format(epoch + 1, args.epochs, mean_loss, save_file))


if __name__ == '__main__':
    train()
