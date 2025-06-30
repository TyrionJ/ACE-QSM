import os
import shutil
import numpy as np
import nibabel as nb
from tqdm import tqdm
from typing import Any
from os.path import isdir, join, exists
from sklearn.model_selection import train_test_split

from utils.helper import say_hi, set_seed
from utils.folder_file_operator import maybe_mkdir, save_json, save_pickle


class Processor:
    def __init__(self, dataset_id, raw_folder, processed_folder, logger=None):
        self.dataset_name = self.get_dataset_name(dataset_id, raw_folder)
        self.raw_dataset = join(raw_folder, self.dataset_name)
        self.processed_dataset = join(processed_folder, self.dataset_name)
        maybe_mkdir(self.processed_dataset)
        self.logger = logger or print
        say_hi(self.logger)

    @staticmethod
    def get_dataset_name(dataset_id, raw_folder):
        assert isdir(raw_folder), "The requested raw data folder could not be found"
        for dataset in os.listdir(raw_folder):
            if f'{dataset_id:03d}' in dataset:
                return dataset
        raise f'The requested dataset {dataset_id} could not be found in sab_raw'

    def process_data(self, img_keys):
        self.logger(' Processing data ...')
        img_folder = join(self.raw_dataset, 'imagesTr')
        lbl_folder = join(self.raw_dataset, 'labelsTr')
        to_dir = join(self.processed_dataset, 'data')
        maybe_mkdir(to_dir)

        for img_key in tqdm(img_keys, desc=' State'):
            echo_nii: Any = nb.load(join(img_folder, f'{img_key}.nii.gz'))
            echo_data = echo_nii.get_fdata().astype(np.float16)
            echo_data = np.transpose(echo_data, (3, 0, 1, 2))
            labl_data = nb.load(join(lbl_folder, f'{img_key}.nii.gz')).get_fdata().astype(np.float16)

            np.save(join(to_dir, f'{img_key}_dat.npy'), echo_data)
            np.save(join(to_dir, f'{img_key}_lbl.npy'), labl_data)
            save_pickle({'affine': echo_nii.affine}, join(to_dir, f'{img_key}.pkl'))

    def split_dataset(self, img_keys):
        self.logger(' Splitting dataset ...')
        split_file = join(self.raw_dataset, 'splits.json')
        if exists(split_file):
            shutil.copy(split_file, join(self.processed_dataset, 'splits.json'))
        else:
            split_file = join(self.processed_dataset, 'splits.json')
            train_keys, test_keys = train_test_split(img_keys, train_size=0.7, random_state=24)
            self.logger(f'  train_size={len(train_keys)}， test_size={len(test_keys)}')
            splits = {'train': train_keys, 'test': test_keys}
            save_json(splits, split_file)
            shutil.copy(split_file, join(self.raw_dataset, 'splits.json'))

    def run(self):
        set_seed()
        self.logger(f'Preprocessing dataset {self.dataset_name} ...')
        img_keys = [i[:-7] for i in sorted(os.listdir(join(self.raw_dataset, 'labelsTr'))) if i.endswith('.nii.gz')]
        img_keys = sorted(img_keys)

        self.process_data(img_keys)
        self.split_dataset(img_keys)
        shutil.copy(join(self.raw_dataset, 'data_info.json'), join(self.processed_dataset, 'data_info.json'))
        self.logger('[DONE] Preprocessing.')
