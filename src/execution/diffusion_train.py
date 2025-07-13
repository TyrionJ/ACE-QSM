import argparse

from scripts.diffusion_trainer import DiffusionTrainer

pf = '../../data/ACE-QSM_processed'
rf = '../../data/ACE-QSM_results'
ds = '0'


def run_trainer():
    parser = argparse.ArgumentParser()

    parser.add_argument('-pre_dir', type=str, default=pf, help='processed folder')
    parser.add_argument('-rst_dir', type=str, default=rf, help='results folder')
    parser.add_argument('-D', type=int, default=1, help='dataset ID')
    parser.add_argument('-T', type=str, default='Task001_DWTDiffusion', help='Task name for the dataset')
    parser.add_argument('-denoiser', type=str, default='DWTDenoiserNet', help='denoiser model name')
    parser.add_argument('-steps', type=int, default=1000, help='sampling timestamps')
    parser.add_argument('-loss', type=str, default='L2', help='sampling timestamps')
    parser.add_argument('--x0', action='store_true', help='compute loss on pred_x0')

    parser.add_argument('-batch', type=int, default=2, help='batch size')
    parser.add_argument('-patch', type=str, default='[64, 64, 48]', help='patch size')
    parser.add_argument('-devices', type=str, default=ds, help='device: cpu or 0, 1, 2, ...')
    parser.add_argument('-iters', type=int, default=650000, help='iteration number')
    parser.add_argument('-sintr', type=int, default=1000, help='save interval')
    parser.add_argument('--c', action='store_true', help='continue train')

    args = parser.parse_args()
    tr = DiffusionTrainer(proposed_dir=args.pre_dir, result_dir=args.rst_dir,
                          dataset_id=args.D, task_name=args.task, denoiser=args.denoiser, steps=args.steps,
                          batch_size=args.batch, patch_size=eval(args.patch), devices=args.devices,
                          iters=args.iters, save_interval=args.sintr, loss_func=args.loss, go_on=args.c, loss_x0=args.x0)
    tr.run()


if __name__ == '__main__':
    run_trainer()
