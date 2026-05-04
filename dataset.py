import numpy as np
from torch.utils.data import Dataset
from PIL import Image, ImageFilter
import cv2
import torch
import os
from torchvision import transforms

from scipy.ndimage.interpolation import rotate
import torch.nn.functional as F


DATASET_ALIASES = {
    'DUTS-TE': ['DUTS-TE', 'DUTS_test'],
    'DUT-OMRON': ['DUT-OMRON', 'DUT_O'],
    'ECSSD': ['ECSSD'],
    'HKU-IS': ['HKU-IS', 'HKU_IS'],
    'PASCAL-S': ['PASCAL-S', 'PASCAL_S'],
    'SOD': ['SOD'],
    'DUTS-TR': ['DUTS-TR'],
}


def resolve_mask_dir(dataset_path):
    dataset_path = os.path.normpath(str(dataset_path))
    dataset_name = os.path.basename(dataset_path)
    aliases = DATASET_ALIASES.get(dataset_name, [dataset_name])
    mask_dir_names = ['mask', 'masks', 'GT', 'gt', 'segmentations', 'SegmentationClass']

    candidate_dirs = []
    for mask_dir_name in mask_dir_names:
        candidate_dirs.append(os.path.join(dataset_path, mask_dir_name))

    for alias in aliases:
        for root in ['dataset/GT', './dataset/GT', 'dataset', './dataset']:
            for mask_dir_name in mask_dir_names:
                candidate_dirs.append(os.path.join(root, alias, mask_dir_name))

    for candidate_dir in candidate_dirs:
        if os.path.isdir(candidate_dir):
            return candidate_dir

    raise FileNotFoundError(
        'Cannot find GT mask directory for {}. Tried: {}'.format(
            dataset_path,
            ', '.join(candidate_dirs[:12]),
        )
    )


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, gt, mask, edge, grays):
        # assert img.size == mask.size
        if img.size == mask.size:
            pass
        else:
            print(img.size, mask.size)

        for t in self.transforms:
            img, gt, mask, edge, grays = t(img, gt, mask, edge, grays)
        return img, gt, mask, edge, grays


class RandomHorizontallyFlip(object):
    def __call__(self, img, gt, mask, edge, grays):
        if np.random.random() < 0.5:
            return img.transpose(Image.FLIP_LEFT_RIGHT), gt.transpose(Image.FLIP_LEFT_RIGHT) , mask.transpose(Image.FLIP_LEFT_RIGHT), edge.transpose(
                Image.FLIP_LEFT_RIGHT), grays.transpose(Image.FLIP_LEFT_RIGHT)
        return img, gt, mask, edge, grays


class JointResize(object):
    def __init__(self, size):
        if isinstance(size, int):
            self.size = (size, size)
        elif isinstance(size, tuple):
            self.size = size
        else:
            raise RuntimeError("size参数请设置为int或者tuple")

    def __call__(self, img, mask):
        img = img.resize(self.size, resample=Image.BILINEAR)
        mask = mask.resize(self.size, resample=Image.NEAREST)
        return img, mask


class RandomRotate(object):
    def __call__(self, img, mask, edge, angle_range=(0, 180)):
        self.degree = np.random.randint(*angle_range)
        rotate_degree = np.random.random() * 2 * self.degree - self.degree
        return img.rotate(rotate_degree, Image.BILINEAR), mask.rotate(rotate_degree, Image.NEAREST), edge.rotate(
            rotate_degree, Image.NEAREST)


class RandomScaleCrop(object):
    def __init__(self, input_size, scale_factor):

        self.input_size = input_size
        self.scale_factor = scale_factor

    def __call__(self, img, mask):
        # random scale (short edge)
        assert img.size[0] == self.input_size

        o_size = np.random.randint(int(self.input_size * 1), int(self.input_size * self.scale_factor))
        img = img.resize((o_size, o_size), resample=Image.BILINEAR)
        mask = mask.resize((o_size, o_size), resample=Image.NEAREST)  #

        # random crop input_size
        x1 = np.random.randint(0, o_size - self.input_size)
        y1 = np.random.randint(0, o_size - self.input_size)
        img = img.crop((x1, y1, x1 + self.input_size, y1 + self.input_size))
        mask = mask.crop((x1, y1, x1 + self.input_size, y1 + self.input_size))

        return img, mask


class ScaleCenterCrop(object):
    def __init__(self, input_size):
        self.input_size = input_size

    def __call__(self, img, mask):
        w, h = img.size
        if w > h:
            oh = self.input_size
            ow = int(1.0 * w * oh / h)
        else:
            ow = self.input_size
            oh = int(1.0 * h * ow / w)
        img = img.resize((ow, oh), resample=Image.BILINEAR)
        mask = mask.resize((ow, oh), resample=Image.NEAREST)

        w, h = img.size
        x1 = int(round((w - self.input_size) / 2.0))
        y1 = int(round((h - self.input_size) / 2.0))
        img = img.crop((x1, y1, x1 + self.input_size, y1 + self.input_size))
        mask = mask.crop((x1, y1, x1 + self.input_size, y1 + self.input_size))

        return img, mask


class RandomGaussianBlur(object):
    def __call__(self, img, mask):
        if np.random.random() < 0.5:
            img = img.filter(ImageFilter.GaussianBlur(radius=np.random.random()))

        return img, mask


class RandomCrop(object):
    def __call__(self, image, gt, mask, edge, grays):
        image = np.array(image)
        gt = np.array(gt)
        mask = np.array(mask)
        edge = np.array(edge)
        grays = np.array(grays)
        H, W, _ = image.shape
        randw = np.random.randint(W / 8)
        randh = np.random.randint(H / 8)
        offseth = 0 if randh == 0 else np.random.randint(randh)
        offsetw = 0 if randw == 0 else np.random.randint(randw)
        p0, p1, p2, p3 = offseth, H + offseth - randh, offsetw, W + offsetw - randw
        if mask is None:
            return image[p0:p1, p2:p3, :]
        image = Image.fromarray(image[p0:p1, p2:p3, :])
        gt    = Image.fromarray(gt[p0:p1, p2:p3].astype('uint8'))
        mask = Image.fromarray(mask[p0:p1, p2:p3].astype('uint8'))
        edge = Image.fromarray(edge[p0:p1, p2:p3].astype('uint8'))
        grays = Image.fromarray(grays[p0:p1, p2:p3].astype('uint8'))

        return image, gt, mask, edge, grays


####################################################################

class Config(object):
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def __getattr__(self, name):
        if name in self.kwargs:
            return self.kwargs[name]
        else:
            return None

class UData(Dataset):
    def __init__(self, cfg):
        self.cfg = cfg
        self.disable_flip = bool(getattr(cfg, 'disable_flip', False))

        self.train_resize = transforms.Resize(size=(352, 352))
        self.to_tensor = transforms.ToTensor()
        self.normalize = transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))

        self.image_transform_test = transforms.Compose([
            transforms.Resize((352, 352)),  
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        self.samples = self._load_samples()
        print(f'dataset getting {len(self.samples)} samples')

    def _load_samples(self):
        primary_txt = os.path.join(self.cfg.datapath, f'{self.cfg.mode}.txt')
        fallback_txt = os.path.join(self.cfg.datapath, 'train.txt') if self.cfg.mode == 'test' else None

        txt_candidates = [primary_txt]
        if fallback_txt is not None and fallback_txt != primary_txt:
            txt_candidates.append(fallback_txt)

        for txt_path in txt_candidates:
            if txt_path is not None and os.path.exists(txt_path):
                with open(txt_path, 'r') as lines:
                    samples = [line.strip() for line in lines if line.strip()]
                print(f'loading samples from {txt_path}')
                return samples

        for image_dir in ['images', 'image']:
            image_root = os.path.join(self.cfg.datapath, image_dir)
            if os.path.isdir(image_root):
                samples = [
                    file_name for file_name in sorted(os.listdir(image_root))
                    if os.path.isfile(os.path.join(image_root, file_name))
                    and os.path.splitext(file_name)[1].lower() in ['.jpg', '.jpeg', '.png', '.bmp']
                ]
                if samples:
                    print(f'loading samples by scanning {image_root}')
                    return samples

        raise FileNotFoundError(
            f'Cannot find sample list for mode={self.cfg.mode} under {self.cfg.datapath}. '
            'Expected train.txt/test.txt or an image/images directory with image files.'
        )

    def _resolve_image_path(self, sample_name):
        normalized_name = sample_name.strip().replace('\\', '/')
        base_name = os.path.basename(normalized_name)
        stem = os.path.splitext(base_name)[0]

        for image_dir in ['images', 'image']:
            for ext in ['.jpg', '.jpeg', '.png', '.bmp']:
                image_path = os.path.join(self.cfg.datapath, image_dir, stem + ext)
                if os.path.exists(image_path):
                    return image_path

        raise FileNotFoundError(
            f'Cannot find image for {sample_name} under {self.cfg.datapath}/images or {self.cfg.datapath}/image'
        )

    def __getitem__(self, idx):
        sample_name = self.samples[idx]
        image_path = self._resolve_image_path(sample_name)
        name = os.path.splitext(os.path.basename(image_path))[0]
        image = Image.open(image_path).convert('RGB')

        if self.cfg.mode == 'train':
            orig_w, orig_h = image.size
            flipped = False
            if (not self.disable_flip) and np.random.random() < 0.5:
                image = image.transpose(Image.FLIP_LEFT_RIGHT)
                flipped = True
            image = self.train_resize(image)
            image = self.to_tensor(image)
            image = self.normalize(image)
            meta = {
                'orig_h': orig_h,
                'orig_w': orig_w,
                'flipped': int(flipped),
            }
            return image, name, meta
        else:
            shape = image.size[::-1]
            image = self.image_transform_test(image)
            return image, shape, name

    def __len__(self):
        return len(self.samples)

if __name__ == '__main__':
    import matplotlib.pyplot as plt

    cfg = Config(mode='train', datapath='./dataset/DUTS/DUTS-TR')
    data = UData(cfg)

    for image, mask, edge in data:
        image = np.array(image).transpose((1, 2, 0))
        mask = np.array(mask).squeeze()
        edge = np.array(edge).squeeze()

        print(image.shape, type(image))

        plt.figure()
        plt.subplot(1, 3, 1)
        plt.imshow(image)  
        plt.subplot(1, 3, 2)
        plt.imshow(mask)
        plt.subplot(1, 3, 3)
        plt.imshow(edge)
        plt.show()
        plt.pause(1)
        # input()
