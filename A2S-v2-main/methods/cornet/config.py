import sys
import argparse  
import os
from base.config import base_config, cfg_convert


def get_config():
    # Default configure
    cfg_dict = {
        'optim': 'SGD',
        'schedule': 'StepLR',
        'lr': 0.005,
        'batch': 16,
        'ave_batch': 1,
        'epoch': 10,
        'step_size': '20,24',
        'gamma': 0.5,
        'clip_gradient': 0,
        'test_batch': 1
    }
    
    parser = base_config(cfg_dict)
    parser.set_defaults(stage=2, trset='c', vals='ce')
    # Add custom params here
    parser.add_argument('--resdual', default=0.4, type=float)
    parser.add_argument('--use-sam-pseudo', action='store_true', help='Run 1 CCAM warmup epoch, generate SAM pseudo labels, then train cornet.')
    parser.add_argument('--sam-checkpoint', default='', help='SAM checkpoint path. Supplying this also enables --use-sam-pseudo.')
    parser.add_argument('--sam-model-type', default='vit_l', help='SAM model type.')
    parser.add_argument('--sam-device', default='cuda', help='Device for SAM inference.')
    parser.add_argument('--sam-score-thresh', default=0.60, type=float)
    parser.add_argument('--sam-heat-iou-thresh', default=0.10, type=float)
    parser.add_argument('--sam-bg-iou-thresh', default=0.15, type=float, help='Maximum SAM candidate IoU with the CCAM background region for normal targets.')
    parser.add_argument('--sam-large-target-bg-iou-thresh', default=0.30, type=float, help='Maximum SAM candidate IoU with the CCAM background region for large/uncertain targets.')
    parser.add_argument('--sam-large-area-thresh', default=0.06, type=float)
    parser.add_argument('--sam-resize-short-edge', default=640, type=int)
    parser.add_argument('--sam-use-crf', action='store_true')
    parser.add_argument(
        '--sam-prompt-mode',
        default='affinity',
        choices=['affinity', 'points'],
        help='affinity: split CCAM prompt with depth/RGB/DINO affinity before SAM; points: use legacy sam_helper_bf.py five-point prompts.',
    )
    parser.add_argument('--sam-depth-root', default='', help='Optional depth map root for CCAM prompt affinity splitting.')
    parser.add_argument('--sam-dino-weight', default='', help='Optional DINOv2 checkpoint for semantic affinity splitting.')
    parser.add_argument('--sam-dino-model', default='dinov2_vitl14', help='DINOv2 torch hub model name.')
    parser.add_argument('--sam-dino-repo', default='', help='Optional local facebookresearch/dinov2 repo path.')
    parser.add_argument('--sam-dino-device', default='', help='Device for DINO inference. Empty means same as --sam-device.')
    parser.add_argument('--sam-dino-max-side', default=700, type=int, help='Maximum long side for DINO inference. 0 means full resolution.')
    parser.add_argument('--sam-dino-pca-dim', default=64, type=int, help='PCA dimension for superpixel DINO descriptors.')
    parser.add_argument('--sam-disable-affinity-split', action='store_true', help='Compatibility alias for --sam-prompt-mode points.')
    parser.add_argument('--sam-seed-points-per-instance', default=3, type=int, help='Max positive SAM points sampled from each affinity instance.')
    parser.add_argument('--sam-disable-neg-prompt', action='store_true', help='Disable conservative background negative points for affinity prompt comparison outputs.')
    parser.add_argument('--sam-neg-ccam-thresh', default=0.25, type=float, help='CCAM threshold used to form connected regions for shared negative point selection.')
    parser.add_argument('--sam-neg-bg-thresh', default=0.05, type=float, help='Maximum CAM value allowed for a negative point candidate.')
    parser.add_argument('--sam-neg-box-expand', default=0.15, type=float, help='Expansion ratio for the CCAM connected-component box used to search negative points.')
    parser.add_argument('--sam-neg-margin', default=8, type=int, help='Pixel margin dilated around all CCAM components and excluded from negative point candidates.')
    parser.add_argument('--sam-neg-points-per-component', default=1, type=int, help='Maximum negative SAM points sampled from each CCAM connected component.')
    parser.add_argument('--sam-mask-prompt-fg-logit', default=3.0, type=float, help='Maximum foreground logit used by soft SAM mask prompts.')
    parser.add_argument('--sam-refine-missing-ratio-thresh', default=0.20, type=float, help='Missing CCAM high-activation ratio threshold for case-B point refine.')
    parser.add_argument(
        '--sam-use-mask-prompt',
        dest='sam_affinity_use_mask_prompt',
        action='store_true',
        help='Also save SAM results with positive mask prompts. Affinity mode saves instance-mask+group-point and whole-mask+group-point results.',
    )
    parser.add_argument(
        '--sam-affinity-use-mask-prompt',
        dest='sam_affinity_use_mask_prompt',
        action='store_true',
        help='Compatibility alias for --sam-use-mask-prompt.',
    )
    parser.add_argument('--sam-pseudo-root', default='./pseudo/cornet_sam', help='Final root used by cornet training after SAM pseudo labels are synced back from SSD staging.')
    parser.add_argument('--sam-pseudo-ssd-parent', default='/tmp/xiao_ssd_data/STC-main/A2S-v2-main/pseudo', help='SSD parent directory used to stage SAM pseudo labels during warmup.')
    parser.add_argument('--sam-pseudo-work-root', default='', help='Optional exact SSD staging root. Empty means derive it from --sam-pseudo-ssd-parent and --sam-pseudo-root.')
    parser.add_argument('--pseudo-root', default='', help='Existing or generated pseudo-label directory used for stage-2 training.')
    parser.add_argument('--ccam-hith', default=0.55, type=float, help='High threshold for CCAM foreground prompt.')
    parser.add_argument('--ccam-loth', default=0.15, type=float, help='Low threshold for CCAM background mask.')
    
    params = parser.parse_args()
    config = vars(params)
    cfg_convert(config)
    print('Training {} network with {} backbone using Gpu: {}'.format(config['model_name'], config['backbone'], config['gpus']))
    
    # Config post-process
    config['use_sam_pseudo'] = bool(config['use_sam_pseudo'] or config['sam_checkpoint'])
    if config.get('sam_disable_affinity_split', False):
        config['sam_prompt_mode'] = 'points'
    config['params'] = [['encoder', config['lr'] / 10], ['decoder', config['lr']]]
    config['lr_decay'] = 0.9
    if config['use_sam_pseudo']:
        config['encoder_strides'] = [1, 2, 2, 1]
        config['decoder_train_layer4_stride'] = 2
    else:
        config['encoder_strides'] = [1, 2, 2, 2]
        config['decoder_train_layer4_stride'] = None
    
    return config, None
