import os
import argparse
import numpy as np
import torch as th
import nibabel as nb
from glob import glob
from typing import Any
from os.path import join, exists

from network import get_net
from utils.helper import say_hi
from utils.evaluation import ssim
from utils.folder_file_operator import load_json
from scripts.diffusion_predictor import Predictor


def main(args):
    say_hi(print)

    final_model = join(args.task_dir, 'checkpoints', 'model-final.pt')
    if not exists(final_model):
        print('Please perform testing after completing the model training.')

    device = th.device(f'cuda:{args.device}')
    model_info = th.load(final_model, map_location=device)
    model = get_net(**{'net_name': model_info['denoiser'], 'cond_channels': model_info['cond_chns']}).to(device)
    model.load_state_dict(model_info['state'])
    model.eval()
    print(f'{model_info["denoiser"]} initialized\n')

    predictor = Predictor(model, model_info['patch_size'], args.batch, device, model_info['steps'], noise=model_info['noise'])

    to_dir = join(args.dataset, args.store_dir)
    os.makedirs(to_dir, exist_ok=True)

    data_dir = join(args.dataset, 'imagesTr')
    labl_dir = join(args.dataset, 'labelsTr')

    if args.data_keys == 'test':
        data_keys = sorted(load_json(join(args.dataset, 'splits.json'))['test'], reverse=args.reverse)
    elif args.data_keys == 'train':
        data_keys = sorted(load_json(join(args.dataset, 'splits.json'))['train'], reverse=args.reverse)
    else:
        data_keys = sorted(args.data_keys.split(','), reverse=args.reverse)
    for N, data_key in enumerate(data_keys):
        print(f"[{N+1}/{len(data_keys)}] Sampling {data_key}...")

        if len(glob(join(str(to_dir), f'{data_key}_*.nii.gz'))) > 0:
            continue

        nii_input: Any = nb.load(join(data_dir, f'{data_key}.nii.gz'))
        nii_label: Any = nb.load(join(labl_dir, f'{data_key}.nii.gz'))
        img_label = nii_label.get_fdata()
        img_input = nii_input.get_fdata()
        img_input = np.transpose(img_input, (3, 0, 1, 2))

        ultimate, noise = predictor.run(img_input, model)

        s = ssim(ultimate, img_label)
        l = np.sqrt(abs(ultimate - img_label).mean())
        to_file = join(str(to_dir), f'{data_key}_{s:.6f}_{l:.6f}.nii.gz')
        nb.Nifti1Image(ultimate, nii_input.affine).to_filename(to_file)

        print(f"{data_key} complete, SSIM={s:.6f}, loss={l:.6f}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-dataset', type=str, default='', help='processed data directory')
    parser.add_argument('-task_dir', type=str, default='', help='result task directory')
    parser.add_argument('-store_dir', type=str, default='', help='prediction directory')
    parser.add_argument('-batch', type=int, default=8, help='batch size')
    parser.add_argument('-device', type=str, default='2', help='device: cpu or 0, 1, 2, ...')
    parser.add_argument('-data_keys', type=str, default='test', help='raw data dir')
    parser.add_argument('--reverse', action='store_true', help='infer all or specific one')

    main(parser.parse_args())
