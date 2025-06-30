import os
import random
import numpy as np
import torch as th
import nibabel as nb
from tqdm import tqdm
from os.path import join
from matplotlib import pyplot as plt

from utils.evaluation import ssim
from utils.gaussian_diffusion import create_gaussian_diffusion
from utils.data_process_utils import get_sliding_window_slicers


class DiffusionSampler:
    def __init__(self, processed_folder, sample_dir, model, patch_size, steps, device):
        self.data_dir = join(processed_folder, 'data')
        self.sample_dir = sample_dir
        self.model = model
        self.patch_size = patch_size
        self.device = device
        self.diffusion = create_gaussian_diffusion(steps=steps)

    def run(self, step):
        noise_dir = join(self.sample_dir, f'step-{step:06d}_noise')
        os.makedirs(noise_dir, exist_ok=True)
        th.save(self.model.state_dict(), join(noise_dir, 'model.pt'))

        data_key = set([i[:-8] for i in os.listdir(self.data_dir) if i.endswith('.npy')])
        data_key = random.sample(data_key, 1)[0]
        img_input = np.load(join(self.data_dir, f'{data_key}_dat.npy'))
        img_label = np.load(join(self.data_dir, f'{data_key}_lbl.npy'))

        margin = 10
        _, x_arr, y_arr, z_arr = np.where(img_input != 0)
        min_x, max_x = x_arr.min(), x_arr.max()
        min_y, max_y = y_arr.min(), y_arr.max()
        min_z, max_z = z_arr.min(), z_arr.max()
        lbs = [min_x + margin, min_y + margin, min_z + margin]
        ubs = [max_x - margin, max_y - margin, max_z - margin]
        img_input = img_input[:, lbs[0]:ubs[0], lbs[1]:ubs[1], lbs[2]:ubs[2]]
        img_input = th.from_numpy(img_input).to(self.device)
        img_label = img_label[lbs[0]:ubs[0], lbs[1]:ubs[1], lbs[2]:ubs[2]]

        slicers = get_sliding_window_slicers(self.patch_size, img_input.shape[1:], 0.6)
        slicers = random.sample(slicers, 16)
        slicer_num = len(slicers)
        results = []
        with th.no_grad():
            for i in tqdm(range(slicer_num), desc='sampling'):
                cond = {'cond': img_input[slicers[i]][None]}
                noise = th.randn((1, 1, *self.patch_size), device=self.device)

                sample = self.diffusion.ddim_sample_loop(
                    self.model,
                    noise.shape,
                    clip_denoised=True,
                    model_kwargs=cond,
                    noise=noise
                ).contiguous()
                # sample = th.rand_like(noise, device=self.device)

                label = img_label[slicers[i][1:]]
                pred = sample[0, 0].cpu().numpy()
                pred[label == 0] = 0

                sim = ssim(pred, label)
                loss = np.sqrt(abs(label[pred != 0] - pred[pred != 0]).mean())
                c_idx = max(sim - loss, 0)

                ipt = cond['cond'][0, 2].cpu().numpy()
                pred = np.transpose(np.array([pred, label, ipt]), [1, 2, 3, 0])
                results.append(pred)
                th.save(noise.cpu(), join(noise_dir, f'noise_step-{step:06d}_idx-{(i + 1):02d}_c-idx-{c_idx:.4f}_s-{sim:.4f}_L-{loss:.4f}.th'))

        results = results[::-1]
        results = [np.vstack(results[i:i + 4]) for i in range(0, 16, 4)]
        results = np.hstack(results)
        nb.Nifti1Image(results, np.eye(4)).to_filename( join(self.sample_dir, f'step-{step:06d}_nii-all.nii.gz'))

        plt.imsave(join(self.sample_dir, f'step-{step:06d}_im2-pred.png'),  np.flipud(np.fliplr(results[:, :, 24, 0].T)), cmap='gray', vmin=-0.15, vmax=0.15)
        plt.imsave(join(self.sample_dir, f'step-{step:06d}_im1-label.png'), np.flipud(np.fliplr(results[:, :, 24, 1].T)), cmap='gray', vmin=-0.15, vmax=0.15)
        plt.imsave(join(self.sample_dir, f'step-{step:06d}_im0-input.png'), np.flipud(np.fliplr(results[:, :, 24, 2].T)), cmap='gray', vmin=-0.15, vmax=0.15)
