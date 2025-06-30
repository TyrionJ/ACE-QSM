import json
import os.path
from batchgenerators.utilities.file_and_folder_operations import save_pickle, load_pickle


__all__ = ['save_pickle', 'load_pickle', 'save_json', 'load_json', 'maybe_mkdir']


def load_json(file):
    with open(file, 'r', encoding='utf-8') as f:
        a = json.load(f)
    return a


def save_json(obj, file, indent=4, sort_keys=True):
    with open(file, 'w') as f:
        json.dump(obj, f, sort_keys=sort_keys, indent=indent, ensure_ascii=False)


def maybe_mkdir(Dir):
    if not os.path.exists(Dir):
        os.makedirs(Dir)
