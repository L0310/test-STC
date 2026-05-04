import math
import os

import torch
from torch import nn
from torch.nn import functional as F


def _resolve_pretrain_path(pretrained):
    if not isinstance(pretrained, str) or pretrained == '':
        return None
    if os.path.isfile(pretrained):
        return pretrained
    search_roots = [os.path.join('..', 'PretrainModel'), os.path.join('.', 'PretrainModel')]
    for root in search_roots:
        base_path = os.path.join(root, pretrained)
        if os.path.isfile(base_path):
            return base_path
        for ext in ('.pth', '.pth.tar', '.tar'):
            candidate = base_path + ext
            if os.path.isfile(candidate):
                return candidate
    return None


def _load_pretrain(model, path):
    checkpoint = torch.load(path, map_location='cpu')
    if isinstance(checkpoint, dict):
        state_dict = checkpoint.get('state_dict', checkpoint.get('model', checkpoint))
    else:
        state_dict = checkpoint

    new_dict = {}
    for k, v in state_dict.items():
        for prefix in ('module.', 'backbone.', 'student.', 'teacher.'):
            if k.startswith(prefix):
                k = k[len(prefix):]
        if k.startswith(('head.', 'fc.')):
            continue
        new_dict[k] = v

    missing, unexpected = model.load_state_dict(new_dict, strict=False)
    print('ViT pretrain loaded: {}, missing: {}, unexpected: {}.'.format(
        len(new_dict), len(missing), len(unexpected)
    ))


def _resize_pos_embed(pos_embed, new_grid_size, num_prefix_tokens):
    if num_prefix_tokens > 0:
        prefix = pos_embed[:, :num_prefix_tokens]
        pos_tokens = pos_embed[:, num_prefix_tokens:]
    else:
        prefix = None
        pos_tokens = pos_embed

    num_pos = pos_tokens.shape[1]
    old_grid = int(math.sqrt(num_pos))
    if old_grid * old_grid != num_pos:
        raise ValueError('Unexpected pos_embed length: {}'.format(num_pos))

    pos_tokens = pos_tokens.reshape(1, old_grid, old_grid, -1).permute(0, 3, 1, 2)
    pos_tokens = F.interpolate(pos_tokens, size=new_grid_size, mode='bicubic', align_corners=False)
    pos_tokens = pos_tokens.permute(0, 2, 3, 1).reshape(1, new_grid_size[0] * new_grid_size[1], -1)

    if prefix is None:
        return pos_tokens
    return torch.cat((prefix, pos_tokens), dim=1)


class ViTEncoder(nn.Module):
    def __init__(self, model_name, pretrained, out_indices):
        super(ViTEncoder, self).__init__()
        try:
            import timm
        except ImportError as exc:
            raise ImportError('timm is required for ViT backbone. Install with `pip install timm`.') from exc

        self.model = None
        model_names = model_name if isinstance(model_name, (list, tuple)) else (model_name,)
        last_err = None
        for name in model_names:
            try:
                self.model = timm.create_model(name, pretrained=False, dynamic_img_size=True)
                break
            except TypeError:
                self.model = timm.create_model(name, pretrained=False)
                break
            except Exception as exc:
                last_err = exc
        if self.model is None:
            raise last_err
        self.out_indices = set(out_indices)
        self.embed_dim = getattr(self.model, 'embed_dim', getattr(self.model, 'num_features', None))
        if self.embed_dim is None:
            raise ValueError('Unable to determine ViT embed_dim.')

        patch_size = getattr(self.model.patch_embed, 'patch_size', 16)
        if isinstance(patch_size, int):
            patch_size = (patch_size, patch_size)
        self.patch_size = patch_size
        self.feat_channels = [self.embed_dim] * 5

        if pretrained and pretrained not in ('none', 'random'):
            path = _resolve_pretrain_path(pretrained)
            if path is None and isinstance(pretrained, str) and 'dino' in pretrained:
                path = _resolve_pretrain_path('dino_vitbase16_pretrain.pth')
            if path is None:
                raise FileNotFoundError('ViT pretrain weight not found.')
            print('Use ViT pretrain: {}.'.format(path))
            _load_pretrain(self.model, path)

    def _update_patch_embed(self, h, w):
        patch_embed = self.model.patch_embed
        h = int(h)
        w = int(w)
        if hasattr(patch_embed, 'strict_img_size'):
            patch_embed.strict_img_size = False
        if hasattr(patch_embed, 'img_size'):
            patch_embed.img_size = (h, w)
        if hasattr(patch_embed, '_img_size'):
            patch_embed._img_size = (h, w)
        grid_size = (h // self.patch_size[0], w // self.patch_size[1])
        if hasattr(patch_embed, 'grid_size'):
            patch_embed.grid_size = grid_size
        if hasattr(patch_embed, 'num_patches'):
            patch_embed.num_patches = grid_size[0] * grid_size[1]

    def _tokens_to_map(self, tokens, h, w, num_prefix_tokens):
        if hasattr(self.model, 'norm') and self.model.norm is not None:
            tokens = self.model.norm(tokens)
        if num_prefix_tokens > 0:
            tokens = tokens[:, num_prefix_tokens:]
        tokens = tokens.reshape(tokens.size(0), h, w, -1).permute(0, 3, 1, 2).contiguous()
        return tokens

    def forward(self, x):
        b, _, h, w = x.shape
        self._update_patch_embed(h, w)
        patch_h = h // self.patch_size[0]
        patch_w = w // self.patch_size[1]

        x = self.model.patch_embed(x)
        if isinstance(x, (tuple, list)):
            x = x[0]
        if x.dim() == 4:
            # Handle both NCHW and NHWC outputs.
            if x.shape[1] != self.embed_dim and x.shape[-1] == self.embed_dim:
                x = x.permute(0, 3, 1, 2).contiguous()
            patch_h, patch_w = x.shape[-2:]
            x = x.flatten(2).transpose(1, 2).contiguous()
        elif x.dim() == 3:
            if x.shape[-1] != self.embed_dim and x.shape[1] == self.embed_dim:
                x = x.transpose(1, 2).contiguous()
            grid_size = getattr(self.model.patch_embed, 'grid_size', None)
            if isinstance(grid_size, (tuple, list)) and len(grid_size) == 2:
                patch_h, patch_w = grid_size
        cls_token = getattr(self.model, 'cls_token', None)
        dist_token = getattr(self.model, 'dist_token', None)
        if cls_token is not None:
            cls_token = cls_token.expand(b, -1, -1)
        if dist_token is not None:
            dist_token = dist_token.expand(b, -1, -1)

        if cls_token is not None and dist_token is not None:
            x = torch.cat((cls_token, dist_token, x), dim=1)
            num_prefix_tokens = 2
        elif cls_token is not None:
            x = torch.cat((cls_token, x), dim=1)
            num_prefix_tokens = 1
        else:
            num_prefix_tokens = 0

        if hasattr(self.model, 'pos_embed') and self.model.pos_embed is not None:
            pos_embed = self.model.pos_embed
            if pos_embed.shape[1] != x.shape[1]:
                pos_embed = _resize_pos_embed(pos_embed, (patch_h, patch_w), num_prefix_tokens)
            x = x + pos_embed

        if hasattr(self.model, 'pos_drop') and self.model.pos_drop is not None:
            x = self.model.pos_drop(x)

        outs = []
        for i, blk in enumerate(self.model.blocks):
            x = blk(x)
            if i in self.out_indices:
                outs.append(x)

        if len(outs) != 3:
            raise ValueError('Expected 3 feature maps, got {}.'.format(len(outs)))

        feat_maps = [self._tokens_to_map(o, patch_h, patch_w, num_prefix_tokens) for o in outs]
        f2, f3, f4 = feat_maps

        # Scheme A: use three intermediate block features directly.
        return [f2, f2, f2, f3, f4]


def vit_base(pretrained='dino_vitbase16_pretrain.pth', out_indices=(2, 5, 8)):
    if isinstance(pretrained, str) and ('vitbase8' in pretrained.lower() or 'patch8' in pretrained.lower()):
        return ViTEncoder(
            (
                'vit_base_patch8_224',
                'vit_base_patch8_224.dino',
                'vit_base_patch8_224_dino',
            ),
            pretrained=pretrained,
            out_indices=out_indices,
        )
    return ViTEncoder('vit_base_patch16_224', pretrained=pretrained, out_indices=out_indices)
