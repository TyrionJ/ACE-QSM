import os
import torch
import torch.distributed as dist


def init_dist():
    if dist.is_initialized():
        return

    world_size = int(os.getenv('WORLD_SIZE', 0))
    rank = int(os.getenv('RANK', 0))
    dist.init_process_group(backend=dist.Backend.GLOO, world_size=world_size, rank=rank)


def get_device(devices):
    devices = [int(i) for i in devices.split(',') if i != 'cpu']
    local_rank = int(os.getenv('LOCAL_RANK', '0'))
    device = torch.device(f'cuda:{devices[local_rank % len(devices)]}') if len(devices) else torch.device('cpu')
    use_dist = len(devices) > 1

    return local_rank, device, use_dist


def sync_dist():
    dist.barrier()


def destroy_process():
    dist.destroy_process_group()
