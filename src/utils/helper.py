import torch
import numpy as np


def say_hi(logger=print):
    logger("\n####################################################################\n"
           "   Please cite the following paper when using ACE-QSM: \n"
           "\n"
           "   He J, Fu B, Wang B, et al. ACE-QSM: Accelerating \n"
           "   quantitative susceptibility mapping acquisition using diffusion \n"
           "   models by reducing repetition time[J]. Biomedical Signal \n"
           "   Processing and Control, 2025, 110: 108217.\n"
           "#####################################################################\n")


def set_seed(seed=12345):
    torch.manual_seed(seed)
    np.random.seed(seed)


def empty_cache(device: torch.device):
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    elif device.type == 'mps':
        from torch import mps
        mps.empty_cache()
    else:
        pass
