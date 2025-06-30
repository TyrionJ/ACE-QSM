import torch
import argparse
import platform

from scripts.normal_trainer import NormTrainer

torch.autograd.set_detect_anomaly(True)

if platform.system().lower() == 'windows':
    pf = r'F:\Data\runtime\ACE-QSM\ACE-QSM_processed'
    rf = r'F:\Data\runtime\ACE-QSM\ACE-QSM_results'
    d = 'cpu'
else:
    pf = '/remote-home/hejj/Data/runtime/ACE-QSM/ACE-QSM_processed'
    rf = '/remote-home/hejj/Data/runtime/ACE-QSM/ACE-QSM_results'
    d = '0'


def run_trainer():
    parser = argparse.ArgumentParser()

    parser.add_argument('-p', type=str, default=pf, help='processed folder')
    parser.add_argument('-r', type=str, default=rf, help='results folder')
    parser.add_argument('-D', type=int, default=5, help='dataset ID')
    parser.add_argument('-T', type=str, default='Task007_BaseUNet', help='Task name for the dataset')
    parser.add_argument('-N', type=str, default='BaseUNet', help='Task network')
    parser.add_argument('-ps', type=str, default='[64, 64, 48]', help='Train patch size')

    parser.add_argument('-d', type=str, default=d, help='device: cpu or 0, 1, 2, ...')
    parser.add_argument('-e', type=int, default=200, help='epoch number')
    parser.add_argument('-b', type=int, default=2, help='batch size')
    parser.add_argument('--c', action='store_true', help='continue train')
    parser.add_argument('--v', action='store_true', help='only validation if train finished')
    args = parser.parse_args()

    tr = NormTrainer(batch_size=args.b,
                    patch_size=eval(args.ps),
                    processed_folder=args.p,
                    dataset_id=args.D,
                    task_name=args.T,
                    net_name=args.N,
                    result_folder=args.r,
                    go_on=args.c,
                    epochs=args.e,
                    device=args.d,
                    validation=args.v,
                    logger=print)
    tr.run()


if __name__ == '__main__':
    run_trainer()
