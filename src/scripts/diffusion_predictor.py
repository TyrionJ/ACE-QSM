import numpy as np
import torch as th
from os.path import exists

from utils.data_process_utils import get_sliding_window_slicers
from utils.gaussian_diffusion import create_gaussian_diffusion


class Predictor:
    def __init__(self, model, patch_size, batch_size, device, timesteps, noise=None, tile_step_size=0.6):
        self.model = model
        self.patch_size = patch_size
        self.batch_size = batch_size
        self.device = device
        self.noise = noise
        self.tile_step_size = tile_step_size
        self.diffusion = create_gaussian_diffusion(steps=timesteps)

    def run(self, img_input, model):
        """
        Predict the fine QSM from QSM_{sTE} via sliding window with fixed patch size.

        @param img_input: 4-D array [echo, W, H, D] with the 1st dimension indicating the echos
        @param model: diffusion noise estimator

        @return: predicted QSM (3-D array)
        """

        margin = 10
        original_shape = img_input.shape[1:]

        _, x_arr, y_arr, z_arr = np.where(img_input != 0)
        min_x, max_x = x_arr.min(), x_arr.max()
        min_y, max_y = y_arr.min(), y_arr.max()
        min_z, max_z = z_arr.min(), z_arr.max()
        lbs = [max(0, min_x - margin), max(0, min_y - margin), max(0, min_z - margin)]
        ubs = [max_x + margin, max_y + margin, max_z + margin]
        img_input = img_input[:, lbs[0]:ubs[0], lbs[1]:ubs[1], lbs[2]:ubs[2]]
        img_input = th.from_numpy(img_input).to(self.device)

        predicted = th.zeros((1, *img_input.shape[1:]), dtype=th.float32, device=self.device)
        n_predictions = th.zeros((1, *img_input.shape[1:]), dtype=th.float32, device=self.device)
        slicers = get_sliding_window_slicers(self.patch_size, img_input.shape[1:], self.tile_step_size)
        slicer_num = len(slicers)

        with th.no_grad():
            if isinstance(self.noise, str) and exists(self.noise):
                noise = th.load(self.noise).to(self.device)
            elif isinstance(self.noise, th.Tensor):
                noise = self.noise.to(self.device)
            else:
                noise = th.randn((1, 1, *self.patch_size), device=self.device)

            noise = th.concat([noise for _ in range(self.batch_size)])

            for i in range(0, slicer_num, self.batch_size):
                print(f'{i}/{slicer_num}')

                cond = []
                for j in range(self.batch_size):
                    if i + j < slicer_num:
                        cond.append(img_input[slicers[i + j]][None])
                mini_batch = len(cond)
                cond = {'cond': th.concat(cond)}

                sample = self.diffusion.ddim_sample_loop(
                    model,
                    (mini_batch, 1, *self.patch_size),
                    clip_denoised=True,
                    model_kwargs=cond,
                    progress=True,
                    noise=noise[:mini_batch]
                ).contiguous()

                for j in range(self.batch_size):
                    if i + j < slicer_num:
                        predicted[slicers[i + j]] += sample[j]
                        n_predictions[slicers[i + j]] += 1

        mask = np.sum(img_input.cpu().numpy(), axis=0)
        mask[mask != 0] = 1
        predicted /= n_predictions
        predicted = predicted[0].cpu().numpy()
        predicted *= mask

        ultimate = np.zeros(original_shape)
        ultimate[lbs[0]:ubs[0], lbs[1]:ubs[1], lbs[2]:ubs[2]] = predicted

        return ultimate, noise[:1].cpu()
