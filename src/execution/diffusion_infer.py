import os
import argparse
import numpy as np
import torch as th
import nibabel as nb
from os.path import join, dirname, exists

from network import get_net
from utils.helper import say_hi
from scripts.diffusion_predictor import Predictor


def fetch_sTE_QSM(sub_dir, method):
    """
    Custom-defined data acquisition rules
    Get input data (C x W x H x D, for C indicating the echos).
    The data acquisition rules can be customized according to your file architecture.

    @param sub_dir: subject data folder
    @param method: dipole inversion method
    @return: input data and Nifti-data affine
    """

    out_dir = join(sub_dir, 'ACE_no-PAT', 'ACE-QSM_out')
    os.makedirs(out_dir, exist_ok=True)

    echo_files = [join(dirname(out_dir), method, f'{method}_e-{i}.nii.gz', ) for i in [1, 2, 3]]
    input_niis = [nb.load(echo_file) for echo_file in echo_files]
    input_data = [input_nii.get_fdata() for input_nii in input_niis]
    input_data = np.array(input_data, dtype=np.float32)
    input_data[input_data == -0] = 0

    return input_data, input_niis[0].affine, out_dir


def main(args):
    say_hi(print)

    final_model = join(args.task_dir, 'checkpoints', 'model-final.pt')
    if not exists(final_model):
        print('Please perform inference after completing the model training.')

    device = th.device(f'cuda:{args.device}')
    model_info = th.load(final_model, map_location=device)

    model = get_net(**{'net_name': model_info['denoiser'], 'cond_channels': model_info['cond_chns']}).to(device)
    model.load_state_dict(model_info['state'])
    model.eval()
    print(f'{model_info["denoiser"]} initialized\n')

    predictor = Predictor(model, model_info['patch_size'], args.batch, device, model_info['steps'], noise=model_info['noise'])

    subs = [i for i in os.listdir(args.data_dir) if args.sub_name == 'ALL' or i in args.sub_name.split(',')]
    subs = sorted(subs, reverse=args.reverse)
    for N, sub in enumerate(subs):
        print(f"[{N+1}/{len(subs)}] Sampling [{sub}]...")

        sub_dir = join(args.data_dir, sub)
        input_data, affine, out_dir = fetch_sTE_QSM(sub_dir, args.method)
        to_file = join(out_dir, f'{args.method}.nii.gz')
        if exists(to_file):
            continue

        ultimate, _ = predictor.run(input_data, model)
        nb.Nifti1Image(ultimate, affine).to_filename(to_file)
        print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-data_dir', type=str, default='', help='raw data dir')
    parser.add_argument('-task_dir', type=str, default='')
    parser.add_argument('-batch', type=int, default=8, help='batch size')
    parser.add_argument('-device', type=str, default='1', help='device: cpu or 0, 1, 2, ...')
    parser.add_argument('-method', type=str, default='iLSQR', help='dipole inversion methods')
    parser.add_argument('-sub_name', type=str, default='ALL', help='infer all or specific one')
    parser.add_argument('--reverse', action='store_true', help='infer all or specific one')

    main(parser.parse_args())
