from torch.utils import data
import torch
import os
from PIL import Image
import numpy as np

# class EvalDataset(data.Dataset):
#     def __init__(self, img_root, label_root):
#         lst_label = sorted(os.listdir(label_root))
#         lst_pred = sorted(os.listdir(img_root))

#         lst = []
#         for name in lst_label:
#             if name in lst_pred:
#                 lst.append(name)

#         self.image_path = list(map(lambda x: os.path.join(img_root, x), lst))
#         self.label_path = list(map(lambda x: os.path.join(label_root, x), lst))
#         # print(self.image_path)
#         # print(self.image_path.sort())
#     def __getitem__(self, item):
#         pred = Image.open(self.image_path[item]).convert('L')
#         gt = Image.open(self.label_path[item]).convert('L')
#         if pred.size != gt.size:
#             pred = pred.resize(gt.size, Image.BILINEAR)
#         pred_np = np.array(pred)
#         pred_np = ((pred_np - pred_np.min()) / (pred_np.max() - pred_np.min()) * 255).astype(np.uint8)
#         pred = Image.fromarray(pred_np)

#         return pred, gt

#     def __len__(self):
#         return len(self.image_path)
    


class EvalDataset(data.Dataset):
    def __init__(self, img_root, label_root):
        lst_label = sorted(os.listdir(label_root))
        lst_pred = sorted(os.listdir(img_root))

        # 对齐文件名
        lst = [name for name in lst_label if name in lst_pred]
        self.image_path = [os.path.join(img_root, x) for x in lst]
        self.label_path = [os.path.join(label_root, x) for x in lst]

    def __getitem__(self, item):
        # ---------- 读取预测与真值 ----------
        pred = Image.open(self.image_path[item]).convert('L')
        gt = Image.open(self.label_path[item]).convert('L')

        # 调整尺寸保持一致
        if pred.size != gt.size:
            pred = pred.resize(gt.size, Image.BILINEAR)

        # ---------- 关键修改部分 ----------
        # 不做任何再归一化，只保留原 [0,255] 结构
        pred_np = np.array(pred).astype(np.float32) / 255.0
        gt_np = np.array(gt).astype(np.float32) / 255.0

        # 转回 Image 以兼容 transforms.ToTensor()
        pred = Image.fromarray((pred_np * 255).astype(np.uint8))
        gt = Image.fromarray((gt_np * 255).astype(np.uint8))

        return pred, gt

    def __len__(self):
        return len(self.image_path)

