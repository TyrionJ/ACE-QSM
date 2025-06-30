import torch
import numpy as np
from functools import lru_cache
from scipy.ndimage import gaussian_filter


def get_sliding_window_slicers(patch_size, image_size, tile_step_size=0.5):
    slicers = []
    steps = compute_steps_for_sliding_window(image_size, patch_size, tile_step_size)
    for sx in steps[0]:
        for sy in steps[1]:
            for sz in steps[2]:
                slicer = tuple(
                    [slice(None), *[slice(si, si + ti) for si, ti in zip((sx, sy, sz), patch_size)]])
                slicers.append(slicer)
    return slicers


def compute_steps_for_sliding_window(image_size, tile_size, tile_step_size):
    assert [i >= j for i, j in zip(image_size, tile_size)], "image size must be as large or larger than patch_size"
    assert 0 < tile_step_size <= 1, 'step_size must be larger than 0 and smaller or equal to 1'

    target_step_sizes_in_voxels = [i * tile_step_size for i in tile_size]
    num_steps = [int(np.ceil((i - k) / j)) + 1 for i, j, k in zip(image_size, target_step_sizes_in_voxels, tile_size)]

    steps = []
    for dim in range(len(tile_size)):
        max_step_value = image_size[dim] - tile_size[dim]
        if num_steps[dim] > 1:
            actual_step_size = max_step_value / (num_steps[dim] - 1)
        else:
            actual_step_size = 99999999999
        steps_here = [int(np.round(actual_step_size * i)) for i in range(num_steps[dim])]
        steps.append(steps_here)
    return steps


def mirror_and_predict(network, x: torch.Tensor) -> torch.Tensor:
    prediction = network(x)
    prediction += torch.flip(network(torch.flip(x, (2,))), (2,))
    prediction += torch.flip(network(torch.flip(x, (3,))), (3,))
    prediction += torch.flip(network(torch.flip(x, (4,))), (4,))
    prediction += torch.flip(network(torch.flip(x, (2, 3))), (2, 3))
    prediction += torch.flip(network(torch.flip(x, (2, 4))), (2, 4))
    prediction += torch.flip(network(torch.flip(x, (3, 4))), (3, 4))
    prediction += torch.flip(network(torch.flip(x, (2, 3, 4))), (2, 3, 4))

    return prediction / 8


@lru_cache(maxsize=2)
def compute_gaussian(tile_size, sigma_scale=1. / 8, value_scaling_factor=1000, dtype=torch.float16,
                     device=torch.device('cuda', 0)):
    tmp = np.zeros(tile_size)
    center_coords = [i // 2 for i in tile_size]
    sigmas = [i * sigma_scale for i in tile_size]
    tmp[tuple(center_coords)] = 1

    gaussian_importance_map = gaussian_filter(tmp, sigmas, 0, mode='constant', cval=0)
    gaussian_importance_map = torch.from_numpy(gaussian_importance_map).type(dtype).to(device)
    gaussian_importance_map = gaussian_importance_map / torch.max(gaussian_importance_map) * value_scaling_factor
    gaussian_importance_map = gaussian_importance_map.type(dtype)
    gaussian_importance_map[gaussian_importance_map == 0] = torch.min(
        gaussian_importance_map[gaussian_importance_map != 0])

    return gaussian_importance_map
