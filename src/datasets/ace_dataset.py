import numpy as np
from tqdm import tqdm
from enum import Enum
from typing import Any
from os.path import join
from torch.utils.data import Dataset
from torch import from_numpy as to_torch


class ACEDatasetType(Enum):
    ALL = 'all'
    TRAIN = 'train'
    TEST = 'test'


class ACEDataset(Dataset):
    def __init__(self, data_folder, patch_size, set_type=ACEDatasetType.TRAIN, preload=False):

        self.data_folder = join(data_folder, 'data')
        self.patch_size = patch_size
        self.total_pixels = np.prod(patch_size)

        if set_type != ACEDatasetType.ALL:
            from batchgenerators.utilities.file_and_folder_operations import load_json
            splits = load_json(join(data_folder, 'splits.json'))
            self.data_keys = sorted(splits[set_type.value])
        else:
            import os
            self.data_keys = sorted([i[:-8] for i in os.listdir(self.data_folder) if i.endswith('_dat.npy')])

        self.image_cache, self.label_cache, self.lbs_ubs_cache = {}, {}, {}
        for key in tqdm(self.data_keys, desc=f'Loading {set_type.value}', disable=not preload):
            if preload:
                self.image_cache[key] = np.load(join(self.data_folder, f'{key}_dat.npy')).astype(float)
                self.label_cache[key] = np.load(join(self.data_folder, f'{key}_lbl.npy')).astype(float)[None]
                self.lbs_ubs_cache[key] = get_bbox(self.image_cache[key], patch_size)[2:]
            else:
                self.image_cache[key] = join(self.data_folder, f'{key}_dat.npy')
                self.label_cache[key] = join(self.data_folder, f'{key}_lbl.npy')
                self.lbs_ubs_cache[key] = patch_size

        data = np.load(join(self.data_folder, f'{self.data_keys[0]}_dat.npy'))
        self.x_shape = [data.shape[0], *patch_size]
        self.y_shape = [1, *patch_size]

    def __len__(self):
        return len(self.data_keys)

    def __getitem__(self, idx):
        key = self.data_keys[idx]

        if type(self.image_cache[key]) == str:
            self.image_cache[key] = np.load(self.image_cache[key]).astype(float)
            self.label_cache[key] = np.load(self.label_cache[key]).astype(float)[None]
            self.lbs_ubs_cache[key] = get_bbox(self.image_cache[key], self.lbs_ubs_cache[key])[2:]

        img_data, lbl_data = self.image_cache[key], self.label_cache[key]
        while True:
            bbox_lbs, bbox_ubs, _, _ = get_bbox(img_data, self.patch_size, *self.lbs_ubs_cache[key])
            bbox_lbs, bbox_ubs = [0, *bbox_lbs], [len(img_data), *bbox_ubs]
            slicer: Any = tuple([slice(i, j) for i, j in zip(bbox_lbs, bbox_ubs)])

            batch_label = lbl_data[slicer]
            if len(np.where(batch_label == 0)[0]) / self.total_pixels > 0.6:
                continue

            return to_torch(img_data[slicer]).float(), to_torch(batch_label).float()


def get_bbox(data, patch_size, lbs=None, ubs=None):
    if lbs is None or ubs is None:
        margin = 20
        _, x_arr, y_arr, z_arr = np.where(data != 0)
        min_x, max_x = x_arr.min(), x_arr.max()
        min_y, max_y = y_arr.min(), y_arr.max()
        min_z, max_z = z_arr.min(), z_arr.max()

        lbs = [max(0, min_x - margin), max(0, min_y - margin), max(0, min_z - margin)]
        ubs = [max(i - j, 0) for i, j in zip([max_x, max_y, max_z], patch_size)]

    bbox_lbs = [np.random.randint(lbs[i], ubs[i] + 1) for i in range(3)]
    bbox_ubs = [bbox_lbs[i] + patch_size[i] for i in range(3)]

    return bbox_lbs, bbox_ubs, lbs, ubs


if __name__ == '__main__':
    """
    Check dataset via plt view
    """
    import os
    import matplotlib.pyplot as plt
    from torch.utils.data import DataLoader

    check_dir = './check_dataset'
    os.makedirs(check_dir, exist_ok=True)

    dr = '/remote-home/hejj/Data/runtime/ACE-QSM/ACE-QSM_processed/Dataset005_SEPIA_sTR'
    trainset: Any = ACEDataset(data_folder=dr, patch_size=[64, 64, 48], preload=False,
                               fold=0, set_type=ACEDatasetType.TRAIN)
    valset: Any = ACEDataset(data_folder=dr, patch_size=[64, 64, 48], preload=True,
                             fold=0, set_type=ACEDatasetType.VAL)

    data_loader = DataLoader(valset, batch_size=16, shuffle=True, num_workers=4)
    def inifit_loop(loader):
        while True:
            for x, y in iter(loader):
                yield x, y
    data_loader = inifit_loop(data_loader)

    N = 0
    while N < 100:
        print(N+1)
        x, y = next(data_loader)

        plt.subplot(1, 2, 1)
        plt.imshow(x[0, -1, :, :, 24].numpy(), cmap='gray', vmin=-0.15, vmax=0.15)
        plt.subplot(1, 2, 2)
        plt.imshow(y[0, 0, :, :, 24].numpy(), cmap='gray', vmin=-0.15, vmax=0.15)
        plt.savefig(join(check_dir, f'{N:05d}.png'))
        N += 1
