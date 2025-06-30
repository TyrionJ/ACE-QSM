import os
import re
import time
import torch
import numpy as np
from glob import glob
from torch import optim
from os.path import join
from datetime import datetime
from torch.utils.data import DataLoader

from network import get_net
from utils.log_mgr import LogMgr
from utils.helper import say_hi, set_seed
from utils.folder_file_operator import load_json
from scripts.diffusion_sampler import DiffusionSampler
from datasets.ace_dataset import ACEDataset, ACEDatasetType
from utils.gaussian_diffusion import create_gaussian_diffusion
from utils.dist_util import init_dist, get_device, sync_dist, destroy_process

set_seed()


def infinite_loop(dataloader):
    while True:
        for x, y in iter(dataloader):
            yield y, {'cond': x}


def seconds_to_dhms(seconds):
    days, seconds = divmod(seconds, 86400)  # 1 day = 86400 seconds
    hours, seconds = divmod(seconds, 3600)  # 1 hour = 3600 seconds
    minutes, _ = divmod(seconds, 60)  # 1 minute = 60 seconds

    str_rst = ''
    if days > 0:
        str_rst += f'{days}D ' if days < 10 else f'{days} '
    str_rst += f'{hours:02d}:{minutes:02d}'

    return str_rst


class DiffusionTrainer:
    data_looper, sample_dir, model_dir, last_model, last_optm, last_ema, log_file, logger = None, '', '', '', '', '', '', None
    processed_folder, cur_iter, model, ema_model, diffusion, optim, log_mgr, sampler = '', 0, None, None, None, None, None, None

    def __init__(self, proposed_dir, result_dir, dataset_id, task_name, denoiser, steps,
                 batch_size, patch_size, devices, iters, save_interval, go_on, loss_func='L2', loss_x0=False, logger=print):
        self.cond_chns = None
        self.T = steps
        self.lr = 1e-4
        self.iters = iters
        self.go_on = go_on
        self.save_interval = save_interval
        self.log_interval = 40
        self.sample_interval = 50000
        self.patch_size = patch_size
        self.batch_size = batch_size
        self.ema_rate = 0.9999
        self.loss_x0 = loss_x0
        self.loss_func = loss_func
        self.denoiser = denoiser
        self.local_rank, self.device, self.use_dist = get_device(devices)
        if self.use_dist:
            init_dist()

        self.install_dir(proposed_dir, result_dir, dataset_id, task_name)
        self.build_logger(logger)
        say_hi(self.logger)
        self.install_data()
        self.install_model()
        self.show_info()
        if self.use_dist:
            sync_dist()

    def show_info(self):
        self.logger('---------------train info.---------------')
        self.logger(f'iter_steps={self.T}')
        self.logger(f'cond_chns={self.cond_chns}')
        self.logger(f'batch_size={self.batch_size}')
        self.logger(f'patch_size={self.patch_size}')
        self.logger(f'current_iter={self.cur_iter}')
        self.logger(f'iterations={self.iters}')
        self.logger(f'save_interval={self.save_interval}')
        self.logger(f'network={self.model.__class__.__name__}')
        self.logger('-----------------------------------------\n')

    def build_logger(self, logger):
        fw = open(self.log_file, 'a', encoding='utf-8')

        def log_fn(content: str, to_file=False):
            if not to_file:
                logger(content)
            fw.write(f'{content}\n')
            fw.flush()

        def pass_fn(_: str):
            pass

        self.logger = log_fn if self.local_rank == 0 else pass_fn
        self.log_mgr = LogMgr(self.logger)

    def install_data(self):
        dataset = ACEDataset(self.processed_folder, self.patch_size, ACEDatasetType.TRAIN, preload=True)
        if self.use_dist:
            from torch.utils.data import DistributedSampler
            train_sampler = DistributedSampler(dataset)
            train_loader = DataLoader(dataset, batch_size=self.batch_size, sampler=train_sampler)
        else:
            train_loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        self.data_looper = infinite_loop(train_loader)

    def install_dir(self, proposed_dir, result_dir, dataset_id, task_name):
        d_name = [i for i in os.listdir(proposed_dir) if f'{dataset_id:03d}' in i]
        assert len(d_name) > 0, f'dataset {dataset_id} not found in {proposed_dir}'

        d_name = d_name[0]
        self.processed_folder = join(proposed_dir, d_name)
        self.cond_chns = load_json(join(self.processed_folder, 'data_info.json'))['in_chns']
        task_dir = str(join(result_dir, d_name, task_name))

        now = datetime.now()
        log_dir = str(join(task_dir, 'logs'))
        os.makedirs(log_dir, exist_ok=True)
        self.log_file = join(log_dir, f'train_{now.strftime("%Y-%m-%d_%H-%M-%S")}.txt')

        self.sample_dir = str(join(task_dir, 'samples'))
        os.makedirs(self.sample_dir, exist_ok=True)

        self.model_dir = str(join(task_dir, 'checkpoints'))
        os.makedirs(self.model_dir, exist_ok=True)

        self.last_model = str(join(self.model_dir, 'model-latest.pt'))
        self.last_optm = str(join(self.model_dir, 'optm-latest.pt'))
        self.last_ema = str(join(self.model_dir, 'ema-latest.pt'))

    def install_model(self):
        self.model = get_net(**{'net_name': self.denoiser, 'cond_channels': self.cond_chns}).to(device=self.device)
        self.ema_model = get_net(**{'net_name': self.denoiser, 'cond_channels': self.cond_chns}).to(device=self.device)
        self.ema_model.load_state_dict(self.model.state_dict())
        self.optim = optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=0.0)

        if self.go_on and os.path.exists(self.last_model):
            model_state = torch.load(self.last_model, map_location=self.device)
            ema_state = torch.load(self.last_ema, map_location=self.device)
            check_state = torch.load(self.last_optm, map_location=self.device)
            self.cur_iter = check_state['iter']
            self.model.load_state_dict(model_state)
            self.ema_model.load_state_dict(ema_state)
            self.optim.load_state_dict(check_state['optm'])

            self.logger('\nContinue training')
            self.logger(f'  {self.last_model} loaded')
            self.logger(f'  cur_iter: {self.cur_iter}\n')
        elif self.go_on:
            self.logger('\nNo checkpoint, Begin a new training\n')

        if self.use_dist:
            from torch.nn.parallel import DistributedDataParallel as DDP
            self.model = DDP(self.model, device_ids=[self.device.index])

        self.diffusion = create_gaussian_diffusion(steps=self.T)
        self.sampler = DiffusionSampler(self.processed_folder, self.sample_dir, self.model, self.patch_size, self.T, self.device)

    def save_model(self, cur_iter, net_model, opt, num_limit=4):
        check_state = {
            'iter': cur_iter,
            'optm': opt.state_dict()
        }

        torch.save(net_model.state_dict(), self.last_model)
        torch.save(net_model.state_dict(), join(self.model_dir, f'model-{cur_iter:06d}.pt'))

        self.update_ema()
        torch.save(self.ema_model.state_dict(), self.last_ema)

        torch.save(check_state, self.last_optm)
        torch.save(check_state, join(self.model_dir, f'optm-{cur_iter:06d}.pt'))

        files = sorted([i for i in os.listdir(self.model_dir) if i.startswith('model-') and 'latest' not in i])
        while len(files) > num_limit:
            model_file = files.pop(0)
            os.remove(join(self.model_dir, model_file))
            os.remove(join(self.model_dir, f'optm-{model_file[6:]}'))

    def anneal_lr(self, cur_iter):
        if cur_iter < self.iters / 4:
            return
        frac_done = cur_iter / self.iters
        lr = self.lr * (1 - frac_done)
        for param_group in self.optim.param_groups:
            param_group["lr"] = max(lr, 5e-6)

    def update_ema(self):
        """
        Update target parameters to be closer to those of source parameters using an exponential moving average.
        """
        target_params = self.ema_model.parameters()
        source_params = self.model.parameters()
        for targ, src in zip(target_params, source_params):
            targ.detach().mul_(self.ema_rate).add_(src, alpha=1 - self.ema_rate)

    def log_loss_dict(self, ts, losses):
        for key, values in losses.items():
            self.log_mgr.logkv_mean(key, values.mean().item())

            for sub_t, sub_loss in zip(ts.cpu().numpy(), values.detach().cpu().numpy()):
                quartile = int(4 * sub_t / self.T)
                self.log_mgr.logkv_mean(f"{key}_q{quartile}", sub_loss)

    def on_train_end(self):
        noise_dirs = glob(join(self.sample_dir, '*_noise'))
        best_idx, bext_noise, best_model = 0, '', ''
        for noise_dir in noise_dirs:
            noise_files = [i for i in os.listdir(noise_dir) if i.startswith('noise')]
            for noise_file in noise_files:
                c_idx = abs(float(re.findall(r'-0.\d+', noise_file)[0]))
                if c_idx > best_idx:
                    best_idx = c_idx
                    bext_noise = join(noise_dir, noise_file)
                    best_model = join(noise_dir, 'model.pt')
        ssim, loss = re.findall(r'-0.\d+', bext_noise)[-2:]
        self.logger(f'\nbest ssim={ssim[1:]}, loss={loss[1:]}')
        final = {'state': torch.load(best_model), 'noise': torch.load(bext_noise),
                 'patch_size': self.patch_size, 'steps': self.T,
                 'denoiser': self.denoiser, 'cond_chns': self.cond_chns}
        torch.save(final, join(self.model_dir, 'model-final.pt'))

    def run(self):
        self.logger('Run training')

        self.model.train()
        start_time = time.time()
        timesteps = np.arange(1, self.T)
        for cur_iter in range(self.cur_iter, self.iters):
            self.optim.zero_grad()

            x_0, cond = next(self.data_looper)
            x_0 = x_0.to(self.device)
            model_kwargs = {k: v.to(self.device) for k, v in cond.items()}
            model_kwargs['loss_func'] = self.loss_func
            if self.loss_x0:
                model_kwargs['loss_x0'] = True

            indices_np = np.random.choice(timesteps, size=(x_0.shape[0],))
            ts = torch.from_numpy(indices_np).long().to(self.device)

            losses = self.diffusion.training_losses(self.model, x_0, ts, model_kwargs)
            self.log_loss_dict(ts, losses)
            loss = losses["loss"].mean()
            loss.backward()

            self.optim.step()
            self.anneal_lr(cur_iter)

            if (cur_iter + 1) % self.log_interval == 0:
                end_time = time.time()
                eta_sed = (self.iters-(cur_iter + 1))*(end_time - start_time) / self.log_interval

                self.log_mgr.logkv('lr', f'{self.optim.param_groups[0]["lr"]:.6f}')
                self.log_mgr.logkv('iters', int(self.iters))
                self.log_mgr.logkv('step', int(cur_iter + 1))
                self.log_mgr.logkv('duration', f'{(end_time - start_time):.2f}s')
                self.log_mgr.logkv('ETA', seconds_to_dhms(int(eta_sed)))
                self.log_mgr.dumpkvs()
                self.logger('')
                start_time = time.time()

            if (cur_iter + 1) % self.save_interval == 0:
                self.save_model(cur_iter + 1, self.model, self.optim)

            if ((cur_iter + 1) >= self.iters // 2 and
                    ((cur_iter + 1) % self.sample_interval == 0 or (cur_iter + 1) == self.iters)):
                self.sampler.run(cur_iter + 1)

        self.on_train_end()

        if self.use_dist:
            destroy_process()
        self.logger('Training end')
