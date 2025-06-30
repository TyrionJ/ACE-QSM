import time
import torch
import os.path
import warnings
import numpy as np
import nibabel as nb
from tqdm import tqdm
from typing import List
from datetime import datetime
from os.path import join, isdir
from torch.cuda.amp import GradScaler
from torch.utils.data import DataLoader
from batchgenerators.utilities.file_and_folder_operations import load_json, save_json, maybe_mkdir_p, load_pickle

from network import get_net
from utils.evaluation import ssim
from net_loss.ssim_l1 import SSIM_L1
from utils.helper import say_hi, set_seed
from utils.polyrescheduler import PolyLRScheduler
from datasets.ace_dataset import ACEDataset, ACEDatasetType
from utils.data_process_utils import get_sliding_window_slicers

warnings.filterwarnings('ignore')


def infinite_loop(dataloader):
    while True:
        for x, y in iter(dataloader):
            yield x, y


class NormTrainer:
    in_chns = out_chns = None
    network = optimizer = lr_scheduler = None
    processed_folder = result_dataset = check_fdr = result_task_folder = valid_fdr = None

    def __init__(self, batch_size, patch_size, processed_folder, dataset_id, task_name, net_name,
                 result_folder, go_on, epochs, device, validation=False, logger=print):

        self.patch_size = patch_size
        self.batch_size = batch_size
        self.go_on = go_on
        self.epochs = epochs
        self.validation = validation
        self.dataset_id = dataset_id
        self.result_folder = result_folder
        self.task_name = task_name
        self.net_name = net_name
        self.device = torch.device(f'cuda:{device}') if device != 'cpu' else torch.device(device)

        self.install_folder(processed_folder)
        self.save_model_info()
        self.logger = self.build_logger(logger)
        say_hi(self.logger)

        self.cur_epoch = 0
        self.best_epoch = 0
        self.early_stop = 40
        self.initial_lr = 1e-2
        self.weight_decay = 3e-5
        self.train_iters = 250
        self.valid_iters = 50
        self.save_interval = 2
        self.best_ssim = 0
        self.train_loader, self.valid_loader = self.get_tr_vd_loader()
        self.grad_scaler = GradScaler() if device != 'cpu' else None
        self.loss_fn = SSIM_L1()
        self.logger(f'Use loss_fn: {type(self.loss_fn).__name__}')

    def build_logger(self, logger):
        now = datetime.now()
        prefix = 'training' if not self.validation else 'validation'
        log_file = join(self.check_fdr, f'{prefix}_log_{now.strftime("%Y-%m-%d_%H-%M-%S")}.txt')
        fw = open(log_file, 'a', encoding='utf-8')

        def log_fn(content):
            logger(content)
            fw.write(f'{content}\n')
            fw.flush()

        return log_fn

    def install_folder(self, processed_folder):
        assert isdir(processed_folder), "The requested processed data folder could not be found"

        d_name = None
        for dataset in os.listdir(processed_folder):
            if f'{self.dataset_id:03d}' in dataset:
                d_name = dataset
                break
        assert d_name is not None, f'The requested dataset {self.dataset_id} could not be found in processed_folder'

        self.processed_folder = join(processed_folder, d_name)
        self.result_task_folder: str = str(join(self.result_folder, d_name, self.task_name))
        self.check_fdr = join(self.result_task_folder, 'checkpoints')
        self.valid_fdr = join(self.result_task_folder, 'validation')
        maybe_mkdir_p(self.check_fdr)
        maybe_mkdir_p(self.valid_fdr)

        data_info = load_json(join(self.processed_folder, 'data_info.json'))
        self.in_chns = data_info['in_chns']
        self.out_chns = data_info['out_chns']

    def save_model_info(self):
        info = {
            'in_chns': self.in_chns,
            'out_chns': self.out_chns,
            'patch_size': self.patch_size
        }
        save_json(info, join(self.result_task_folder, 'model_info.json'))

    def get_tr_vd_loader(self):
        tr_dataset = ACEDataset(self.processed_folder, self.patch_size, ACEDatasetType.TRAIN, preload=not self.validation)
        vd_dataset = ACEDataset(self.processed_folder, self.patch_size, ACEDatasetType.TEST, preload=not self.validation)
        self.logger(f"tr_set-size={len(tr_dataset)}, val_set-size={len(vd_dataset)}")
        tr_loader = DataLoader(tr_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.batch_size)
        vd_loader = DataLoader(vd_dataset, batch_size=2, shuffle=True, num_workers=2)

        return infinite_loop(tr_loader), infinite_loop(vd_loader)

    def initialize(self, net_name):
        kwargs = {'in_channels': self.in_chns, 'out_channels': self.out_chns}
        self.network = get_net(net_name, **kwargs).to(self.device)
        self.optimizer = torch.optim.AdamW(self.network.parameters(), self.initial_lr, weight_decay=self.weight_decay)
        self.lr_scheduler = PolyLRScheduler(self.optimizer, self.initial_lr, self.epochs)
        self.load_states()

    def load_states(self):
        check_file = join(self.check_fdr, 'check-latest.pt')
        if self.go_on or self.validation:
            if os.path.isfile(check_file):
                self.logger(f'Use checkpoint: {check_file}')
                weights = torch.load(join(self.check_fdr, 'model-latest.pt'), map_location=torch.device('cpu'))
                checkpoint = torch.load(check_file, map_location=torch.device('cpu'))

                self.network.load_state_dict(weights)
                self.optimizer.load_state_dict(checkpoint['optimizer_state'])
                self.cur_epoch = checkpoint['cur_epoch']
                self.best_ssim = checkpoint['best_ssim']
                self.best_epoch = checkpoint['best_epoch']
                if self.grad_scaler is not None and checkpoint['grad_scaler_state'] is not None:
                    self.grad_scaler.load_state_dict(checkpoint['grad_scaler_state'])
            else:
                self.logger('No checkpoint found, start a new training')

    def save_states(self, val_ssim, num_limit=4):
        self.cur_epoch += 1

        if self.best_ssim < val_ssim:
            self.best_ssim = val_ssim
            self.best_epoch = self.cur_epoch
            self.logger(f'Eureka!!! Best ssim: {self.best_ssim:.6f}')
        else:
            self.logger(f'The best epoch: {self.best_epoch}')
            self.logger(f'The best ssim: {self.best_ssim:.6f}')
        self.logger('')

        checkpoint = {
            'optimizer_state': self.optimizer.state_dict(),
            'cur_epoch': self.cur_epoch,
            'best_epoch': self.best_epoch,
            'best_ssim': self.best_ssim,
            'grad_scaler_state': self.grad_scaler.state_dict() if self.grad_scaler is not None else None
        }
        torch.save(self.network.state_dict(), join(self.check_fdr, 'model-latest.pt'))
        torch.save(checkpoint, join(self.check_fdr, 'check-latest.pt'))

        torch.save(self.network.state_dict(), join(self.check_fdr, f'model-{val_ssim:.6f}.pt'))
        torch.save(checkpoint, join(self.check_fdr, f'check-{val_ssim:.6f}.pt'))

        files = sorted([i for i in os.listdir(self.check_fdr) if i.startswith('model-') and 'latest' not in i])
        while len(files) > num_limit:
            model_file = files.pop(0)
            os.remove(join(self.check_fdr, model_file))
            os.remove(join(self.check_fdr, f'check-{model_file[6:]}'))

    def train_step(self, batch):
        self.optimizer.zero_grad()

        in_data, tgt_data = [i.to(self.device) for i in batch]
        net_out = self.network(in_data)
        # if any(tensor([any(tensor([isnan(m).any() for m in net_out[0][i]])) for i in range(len(net_out))])):
        #     return None

        t_loss = self.loss_fn(net_out, tgt_data)
        if torch.isnan(t_loss):
            return None
        try:
            if self.grad_scaler is not None:
                self.grad_scaler.scale(t_loss).backward()
                self.grad_scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
                self.grad_scaler.step(self.optimizer)
                self.grad_scaler.update()
            else:
                t_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
                self.optimizer.step()
        except RuntimeError:
            return None
        return {'loss': t_loss.detach().cpu().numpy()}

    def valid_step(self, batch):
        in_data, tgt_data = [i.to(self.device) for i in batch]

        net_out = self.network(in_data)
        t_loss = self.loss_fn(net_out, tgt_data)
        if torch.isnan(t_loss):
            return None
        vd_ssim = np.mean([ssim(out[0], tgt[0]) for out, tgt in zip(net_out.cpu().numpy(), tgt_data.cpu().numpy())])

        return {'loss': t_loss.cpu().numpy().item(), 'ssim': vd_ssim}

    def on_train_epoch_end(self, epoch, train_loss, lr):
        self.logger(f'Epoch: {epoch} / {self.epochs}')
        self.logger(f'current lr: {np.round(lr, decimals=5)}')
        self.logger(f'train loss: {np.round(train_loss, decimals=6)}')

    def on_valid_epoch_end(self, val_outputs: List[dict]):
        val_outputs = list(filter(None, val_outputs))

        loss_here = np.mean([out['loss'] for out in val_outputs])
        ssim_here = np.mean([out['ssim'] for out in val_outputs])
        self.logger(f'validation loss: {loss_here:.6f}')
        self.logger(f'valid mean ssim: {ssim_here:.6f}')

        return ssim_here

    def conduct_final_testing(self):
        self.logger('\nFinal Validation')

        if self.validation:
            files = sorted([i for i in os.listdir(self.check_fdr) if i.startswith('model-') and 'latest' not in i])
            weights = torch.load(join(self.check_fdr, files[-1]), map_location=self.device)
            self.network.load_state_dict(weights)
            print(f'Load best model: {files[-1]}')

        obj = {}
        mean_ssim = 0
        test_indices = load_json(join(self.processed_folder, 'splits.json'))[ACEDatasetType.TEST.value]
        self.logger(f'There {len(test_indices)} case(s) to validate:\n')
        for N, key in enumerate(test_indices):
            self.logger(f'[{N+1}/{len(test_indices)}] Testing {key}:')
            obj[key] = {}
            in_data = np.load(join(self.processed_folder, 'data', f'{key}_dat.npy'))
            out_tgt = np.load(join(self.processed_folder, 'data', f'{key}_lbl.npy'))
            pkl_info = load_pickle(join(self.processed_folder, 'data', f'{key}.pkl'))

            in_data = torch.from_numpy(in_data).float()
            predicted_qsm = torch.zeros((self.out_chns, *in_data.shape[1:]), dtype=torch.half, device=self.device)
            n_predictions = torch.zeros(in_data.shape[1:], dtype=torch.half, device=self.device)
            slicers = get_sliding_window_slicers(self.patch_size, in_data.shape[1:], tile_step_size=0.6)
            with torch.no_grad():
                for sli in tqdm(slicers, desc='  State'):
                    workon = in_data[sli].to(self.device)
                    prediction = self.network(workon[None])
                    predicted_qsm[sli] += prediction[0]
                    n_predictions[sli[1:]] += 1

            predicted_qsm /= n_predictions
            predicted_qsm[in_data.mean(0)[None] == 0] = 0
            predicted_qsm = predicted_qsm[0].cpu().numpy()
            v_ssim = ssim(predicted_qsm, out_tgt)
            mean_ssim = (mean_ssim * N + v_ssim) / (N + 1)

            self.logger(f'  ssim={v_ssim:.6f}')

            obj[key]['ssim'] = v_ssim

            to_file = join(self.valid_fdr, f'{key}.nii.gz')
            nb.Nifti1Image(predicted_qsm.astype(np.float32), pkl_info['affine']).to_filename(to_file)
            self.logger(f'  Prediction saved\n')

        self.logger('Final Validation complete')
        self.logger(f'  Mean Validation Dice: {np.round(mean_ssim, decimals=6)}')
        obj['mean_dice'] = mean_ssim
        save_json(obj, join(self.valid_fdr, 'validation_summary.json'))

    def run(self):
        set_seed()
        self.initialize(self.net_name)

        if not self.validation:
            self.logger('\nBegin training ...')
            time.sleep(0.5)

            for epoch in range(self.cur_epoch, self.epochs):
                avg_loss = 0

                if epoch - self.best_epoch >= self.early_stop:
                    self.logger('Early stop')
                    break
                self.lr_scheduler.step(self.cur_epoch)
                lr = self.optimizer.param_groups[0]['lr']

                self.network.train()
                with tqdm(desc=f'[{epoch + 1}/{self.epochs}]Training', total=self.train_iters) as p:
                    for batch_id in range(self.train_iters):
                        train_loss = self.train_step(next(self.train_loader))
                        train_loss = train_loss['loss'] if train_loss is not None else avg_loss
                        avg_loss = (avg_loss * batch_id + train_loss) / (batch_id + 1)
                        p.set_postfix(**{'avg': '%.5f' % avg_loss,
                                         'bat': '%.5f' % train_loss,
                                         'lr': '%.5f' % lr,
                                         'Tsk': self.task_name})
                        p.update()

                self.network.eval()
                with torch.no_grad():
                    with tqdm(desc='~~Validation', total=self.valid_iters, colour='green') as p:
                        val_outputs = []
                        for batch_id in range(self.valid_iters):
                            val_outputs.append(self.valid_step(next(self.valid_loader)))
                            p.update()

                self.on_train_epoch_end(epoch + 1, avg_loss, lr)
                val_ssim = self.on_valid_epoch_end(val_outputs)
                self.save_states(val_ssim)
            self.logger('Training end!')

        self.conduct_final_testing()
