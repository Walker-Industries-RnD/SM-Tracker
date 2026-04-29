#!/usr/bin/env python3
"""
Transplant a checkpoint into a model with a redone diameter head.
Compatible keys (matching name + shape) are loaded; mismatched keys
(e.g. diameter_pred after a head redesign) are left at random init.
The optimizer state is dropped since it references stale parameter tensors.

Usage:
    python transplant_checkpoint.py                    # auto-finds latest checkpoint
    python transplant_checkpoint.py --ckpt path/to.pth
    python transplant_checkpoint.py --out transplanted.pth
"""

import argparse
import os
import torch
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights
from model import ShinraCNN


def find_latest_checkpoint():
    phase_dirs = sorted(
        [d for d in os.listdir('.') if d.startswith('phase_') and os.path.isdir(d)],
        key=lambda d: int(d.split('_')[1]),
        reverse=True,
    )
    for phase_dir in phase_dirs:
        checkpoints = sorted(
            [f for f in os.listdir(phase_dir)
             if f.startswith('shinra_checkpoint_') and f.endswith('.pth')],
            key=lambda f: int(f.split('_')[-1].split('.')[0]),
        )
        if checkpoints:
            return os.path.join(phase_dir, checkpoints[-1])
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', default=None)
    parser.add_argument('--out',  default='transplanted_checkpoint.pth')
    args = parser.parse_args()

    ckpt_path = args.ckpt or find_latest_checkpoint()
    if ckpt_path is None:
        raise FileNotFoundError('No checkpoint found.')
    print(f'Source: {ckpt_path}')

    backbone = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.DEFAULT)
    model = ShinraCNN(backbone, out_channels=17)
    new_sd = model.state_dict()

    ckpt = torch.load(ckpt_path, map_location='cpu')
    old_sd = ckpt['shinra']

    compatible   = {k: v for k, v in old_sd.items()
                    if k in new_sd and v.shape == new_sd[k].shape}
    shape_differ = [k for k in old_sd if k in new_sd and old_sd[k].shape != new_sd[k].shape]
    removed      = [k for k in old_sd if k not in new_sd]
    new_keys     = [k for k in new_sd if k not in old_sd and k not in ('grid_x', 'grid_y')]

    compatible['grid_x'] = new_sd['grid_x']
    compatible['grid_y'] = new_sd['grid_y']

    print(f'\n  loaded   : {len(compatible)} keys')
    print(f'  injected : grid_x, grid_y  (soft-argmax coordinate buffers)')
    print(f'  shape mismatch (random init) : {shape_differ}')
    print(f'  removed from model           : {removed}')
    print(f'  new in model (random init)   : {new_keys}')

    model.load_state_dict(compatible, strict=False)

    torch.save({'shinra': model.state_dict(), 'epoch': ckpt['epoch']}, args.out)
    print(f'\nSaved → {args.out}')


if __name__ == '__main__':
    main()
