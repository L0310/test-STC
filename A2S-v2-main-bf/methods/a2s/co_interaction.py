import torch
from torch import nn
from torch.nn import functional as F


def foreground_sign(pred):
    b, c, w, h = pred.size()
    p = pred.gt(0).float()
    num_pos = p[:, :, 0, 0] + p[:, :, w - 1, 0] + p[:, :, w - 1, h - 1] + p[:, :, 0, h - 1]
    sign = ((num_pos < 2).float() * 2 - 1).view(b, c, 1, 1)
    return sign


def normalize(x):
    center = torch.mean(x, dim=(2, 3), keepdim=True)
    return x - center


def _group_count(channels):
    for groups in (32, 16, 8, 4, 2):
        if channels % groups == 0:
            return groups
    return 1


def _conv_gn_relu(cin, cout, kernel_size=3, padding=1):
    return nn.Sequential(
        nn.Conv2d(cin, cout, kernel_size, padding=padding, bias=False),
        nn.GroupNorm(_group_count(cout), cout),
        nn.ReLU(inplace=True),
    )


def _head_num(dim, heads):
    heads = max(1, int(heads))
    for cur in range(min(dim, heads), 0, -1):
        if dim % cur == 0:
            return cur
    return 1


def _boundary_band(prob, kernel_size=5):
    k = max(3, int(kernel_size))
    if k % 2 == 0:
        k += 1
    pad = k // 2
    mask = prob.gt(0.5).float()
    dil = F.max_pool2d(mask, kernel_size=k, stride=1, padding=pad)
    ero = 1 - F.max_pool2d(1 - mask, kernel_size=k, stride=1, padding=pad)
    return (dil - ero).clamp_(0, 1)


def _unwrap_model(model):
    return model.module if hasattr(model, 'module') else model


def _make_scale_adapters(in_dim, out_dim, num_scales=3):
    layers = []
    for _ in range(num_scales):
        if in_dim == out_dim:
            layers.append(nn.Identity())
        else:
            layers.append(_conv_gn_relu(in_dim, out_dim, kernel_size=1, padding=0))
    return nn.ModuleList(layers)


class A2SFuseHead(nn.Module):
    def __init__(self, feat_dim):
        super(A2SFuseHead, self).__init__()
        self.fuse = _conv_gn_relu(feat_dim * 3, feat_dim)
        self.se = nn.Conv2d(feat_dim, feat_dim, 1)

    def forward(self, feats, out_size):
        feat = self.fuse(torch.cat(feats, dim=1))
        feat = torch.sigmoid(self.se(F.adaptive_avg_pool2d(feat, 1))) * feat
        feat = normalize(feat)
        pred = torch.sum(feat, dim=1, keepdim=True)
        pred = pred * foreground_sign(pred)
        pred = F.interpolate(pred, size=out_size, mode='bilinear', align_corners=True)
        return {
            'feat': [feat],
            'sal': [pred],
            'final': pred,
        }


class SpatialGatedFusion(nn.Module):
    def __init__(self, feat_dim, prefer='a'):
        super(SpatialGatedFusion, self).__init__()
        self.prefer = prefer
        self.weight = nn.Sequential(
            _conv_gn_relu(feat_dim * 2 + 3, feat_dim),
            nn.Conv2d(feat_dim, 2, 1),
        )
        self.cross = nn.Sequential(
            _conv_gn_relu(feat_dim * 2, feat_dim),
            _conv_gn_relu(feat_dim, feat_dim),
        )
        self.out = _conv_gn_relu(feat_dim * 3, feat_dim)
        self.bias = nn.Parameter(torch.tensor(1.0))

    def forward(self, feat_a, feat_b, conf_a, conf_b, prior):
        logits = self.weight(torch.cat([feat_a, feat_b, conf_a, conf_b, prior], dim=1))
        if self.prefer == 'a':
            logits[:, 0:1] = logits[:, 0:1] + self.bias * prior
        else:
            logits[:, 1:2] = logits[:, 1:2] + self.bias * prior
        weights = torch.softmax(logits, dim=1)
        cross = self.cross(torch.cat([feat_a, feat_b], dim=1))
        fused = self.out(torch.cat([weights[:, 0:1] * feat_a, weights[:, 1:2] * feat_b, cross], dim=1))
        return fused


class AttentionGatedFusion(nn.Module):
    def __init__(self, feat_dim, shared_dim=64, num_heads=4, attn_max_size=24):
        super(AttentionGatedFusion, self).__init__()
        shared_dim = int(shared_dim)
        num_heads = _head_num(shared_dim, num_heads)
        self.attn_max_size = int(attn_max_size)
        self.proj_a = nn.Conv2d(feat_dim, shared_dim, 1, bias=False)
        self.proj_b = nn.Conv2d(feat_dim, shared_dim, 1, bias=False)
        self.norm_q = nn.LayerNorm(shared_dim)
        self.norm_kv = nn.LayerNorm(shared_dim)
        self.attn_ab = nn.MultiheadAttention(shared_dim, num_heads, batch_first=True)
        self.attn_ba = nn.MultiheadAttention(shared_dim, num_heads, batch_first=True)
        self.restore_a = _conv_gn_relu(shared_dim, feat_dim, kernel_size=1, padding=0)
        self.restore_b = _conv_gn_relu(shared_dim, feat_dim, kernel_size=1, padding=0)
        self.weight = nn.Sequential(
            _conv_gn_relu(feat_dim * 4 + 3, feat_dim),
            nn.Conv2d(feat_dim, 2, 1),
        )
        self.out = nn.Sequential(
            _conv_gn_relu(feat_dim * 4, feat_dim),
            _conv_gn_relu(feat_dim, feat_dim),
        )

    def _cross_attention(self, query_map, context_map, attn):
        b, c, h, w = query_map.shape
        query = query_map.flatten(2).transpose(1, 2).contiguous()
        context = context_map.flatten(2).transpose(1, 2).contiguous()
        query = self.norm_q(query)
        context = self.norm_kv(context)
        out, _ = attn(query, context, context, need_weights=False)
        return out.transpose(1, 2).reshape(b, c, h, w).contiguous()

    def forward(self, feat_a, feat_b, conf_a, conf_b, prior):
        target_size = feat_a.shape[-2:]
        attn_h = min(feat_a.shape[-2], feat_b.shape[-2], self.attn_max_size)
        attn_w = min(feat_a.shape[-1], feat_b.shape[-1], self.attn_max_size)
        attn_size = (attn_h, attn_w)

        feat_a_attn = feat_a if feat_a.shape[-2:] == attn_size else F.adaptive_avg_pool2d(feat_a, attn_size)
        feat_b_attn = feat_b if feat_b.shape[-2:] == attn_size else F.adaptive_avg_pool2d(feat_b, attn_size)

        proj_a = self.proj_a(feat_a_attn)
        proj_b = self.proj_b(feat_b_attn)
        cross_a = self.restore_a(self._cross_attention(proj_a, proj_b, self.attn_ab))
        cross_b = self.restore_b(self._cross_attention(proj_b, proj_a, self.attn_ba))
        if cross_a.shape[-2:] != target_size:
            cross_a = F.interpolate(cross_a, size=target_size, mode='bilinear', align_corners=False)
        if cross_b.shape[-2:] != target_size:
            cross_b = F.interpolate(cross_b, size=target_size, mode='bilinear', align_corners=False)

        logits = self.weight(torch.cat([feat_a, feat_b, cross_a, cross_b, conf_a, conf_b, prior], dim=1))
        weights = torch.softmax(logits, dim=1)
        mixed = weights[:, 0:1] * (feat_a + cross_a) + weights[:, 1:2] * (feat_b + cross_b)
        fused = self.out(torch.cat([mixed, cross_a, cross_b, feat_a + feat_b], dim=1))
        return fused


class MultiScaleComplementaryFusion(nn.Module):
    def __init__(self, feat_dim, backbone_a, backbone_b, shared_dim=128, num_heads=4):
        super(MultiScaleComplementaryFusion, self).__init__()
        a_is_vit = 'vit' in backbone_a.lower()
        b_is_vit = 'vit' in backbone_b.lower()
        if a_is_vit and not b_is_vit:
            local_prefer = 'b'
            semantic_prefer = 'a'
        elif b_is_vit and not a_is_vit:
            local_prefer = 'a'
            semantic_prefer = 'b'
        else:
            local_prefer = 'a'
            semantic_prefer = 'b'

        self.local = SpatialGatedFusion(feat_dim, prefer=local_prefer)
        self.middle = AttentionGatedFusion(feat_dim, shared_dim=min(shared_dim, feat_dim), num_heads=num_heads)
        self.semantic = SpatialGatedFusion(feat_dim, prefer=semantic_prefer)

    def forward(self, feats_a, feats_b, prob_a, prob_b):
        edge = _boundary_band(0.5 * (prob_a + prob_b))
        interior = 1.0 - edge
        agreement = 1.0 - torch.abs(prob_a - prob_b)
        conf_a = torch.abs(prob_a - 0.5)
        conf_b = torch.abs(prob_b - 0.5)

        fused = []
        priors = (edge, agreement, interior)
        mixers = (self.local, self.middle, self.semantic)
        for feat_a, feat_b, prior, mixer in zip(feats_a, feats_b, priors, mixers):
            target_size = (
                max(feat_a.shape[-2], feat_b.shape[-2]),
                max(feat_a.shape[-1], feat_b.shape[-1]),
            )
            if feat_a.shape[-2:] != target_size:
                feat_a = F.interpolate(feat_a, size=target_size, mode='bilinear', align_corners=False)
            if feat_b.shape[-2:] != target_size:
                feat_b = F.interpolate(feat_b, size=target_size, mode='bilinear', align_corners=False)
            prior_s = F.interpolate(prior, size=target_size, mode='bilinear', align_corners=False)
            conf_a_s = F.interpolate(conf_a, size=target_size, mode='bilinear', align_corners=False)
            conf_b_s = F.interpolate(conf_b, size=target_size, mode='bilinear', align_corners=False)
            fused.append(mixer(feat_a, feat_b, conf_a_s, conf_b_s, prior_s))
        return fused


class InteractiveCoA2S(nn.Module):
    def __init__(
        self,
        model_a,
        model_b,
        feat_dim_a,
        feat_dim_b,
        shared_dim=128,
        num_heads=4,
        dropout=0.0,
        detach_branches=True,
        refine_branches=False,
        backbone_a='resnet',
        backbone_b='vit',
    ):
        super(InteractiveCoA2S, self).__init__()
        self.model_a = model_a
        self.model_b = model_b
        self.detach_branches = detach_branches
        self.refine_branches = refine_branches
        feat_dim = min(feat_dim_a, feat_dim_b)
        self.adapt_a = _make_scale_adapters(feat_dim_a, feat_dim)
        self.adapt_b = _make_scale_adapters(feat_dim_b, feat_dim)
        self.msfusion = MultiScaleComplementaryFusion(
            feat_dim=feat_dim,
            backbone_a=backbone_a,
            backbone_b=backbone_b,
            shared_dim=shared_dim,
            num_heads=num_heads,
        )
        self.fuse_head = A2SFuseHead(feat_dim)

    def interaction_parameters(self):
        return (
            list(self.adapt_a.parameters()) +
            list(self.adapt_b.parameters()) +
            list(self.msfusion.parameters()) +
            list(self.fuse_head.parameters())
        )

    def _forward_branch(self, model, x_tensor, phase):
        target = _unwrap_model(model)
        x_size = x_tensor.shape[-2:]
        xs = target.encoder(x_tensor)
        feats = target.decoder.extract_multi_scale(xs)
        out = target.decoder.predict_from_multi_scale(feats, x_size)
        return out, feats

    def forward_branches(self, x, phase='test'):
        x_tensor = x[0] if isinstance(x, (list, tuple)) else x
        out_a, feats_a = self._forward_branch(self.model_a, x_tensor, phase)
        out_b, feats_b = self._forward_branch(self.model_b, x_tensor, phase)

        prob_a = torch.sigmoid(out_a['final'].detach() if self.detach_branches else out_a['final'])
        prob_b = torch.sigmoid(out_b['final'].detach() if self.detach_branches else out_b['final'])
        src_a = [feat.detach() if self.detach_branches else feat for feat in feats_a]
        src_b = [feat.detach() if self.detach_branches else feat for feat in feats_b]
        src_a = [adapter(feat) for adapter, feat in zip(self.adapt_a, src_a)]
        src_b = [adapter(feat) for adapter, feat in zip(self.adapt_b, src_b)]

        fused_feats = self.msfusion(src_a, src_b, prob_a, prob_b)
        out_fuse = self.fuse_head(fused_feats, x_tensor.shape[-2:])
        out_fuse['branch_a_final'] = out_a['final']
        out_fuse['branch_b_final'] = out_b['final']
        out_fuse['branch_a_prob'] = prob_a
        out_fuse['branch_b_prob'] = prob_b
        return out_a, out_b, out_fuse

    def forward(self, x, phase='test'):
        _, _, out_fuse = self.forward_branches(x, phase)
        return out_fuse


def infer_decoder_feat_dim(model):
    target = _unwrap_model(model)
    decoder = getattr(target, 'decoder', None)
    fusion = getattr(decoder, 'fusion', None)
    if fusion is None or not hasattr(fusion, 'ad0'):
        raise ValueError('Unable to infer decoder feature dim for interaction module.')
    conv = fusion.ad0[0]
    return conv.out_channels
