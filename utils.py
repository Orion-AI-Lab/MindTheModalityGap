import logging
import os
import pickle
import random

import numpy as np
import torch
import torch.distributed as dist

logger = logging.getLogger(__name__)


def set_seed(seed: int = 42) -> None:
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)
    logger.info(f"Random seed set as {seed}")


def is_global_master(args):
    return args.rank == 0


def is_local_master(args):
    return args.local_rank == 0


def is_master(args, local=False):
    return is_local_master(args) if local else is_global_master(args)


def is_distributed():
    if "WORLD_SIZE" in os.environ:
        return int(os.environ["WORLD_SIZE"]) > 1
    if "SLURM_NTASKS" in os.environ:
        return int(os.environ["SLURM_NTASKS"]) > 1
    return False


def world_info_from_env():
    local_rank = 0
    for v in (
        "LOCAL_RANK",
        "MPI_LOCALRANKID",
        "SLURM_LOCALID",
        "OMPI_COMM_WORLD_LOCAL_RANK",
    ):
        if v in os.environ:
            local_rank = int(os.environ[v])
            break
    global_rank = 0
    for v in ("RANK", "PMI_RANK", "SLURM_PROCID", "OMPI_COMM_WORLD_RANK"):
        if v in os.environ:
            global_rank = int(os.environ[v])
            break
    world_size = 1
    for v in ("WORLD_SIZE", "PMI_SIZE", "SLURM_NTASKS", "OMPI_COMM_WORLD_SIZE"):
        if v in os.environ:
            world_size = int(os.environ[v])
            break

    return local_rank, global_rank, world_size


def init_distributed_device(args):
    args.distributed = False
    args.world_size = 1
    args.rank = 0
    args.local_rank = 0
    if is_distributed():
        if "SLURM_PROCID" in os.environ:
            args.local_rank, args.rank, args.world_size = world_info_from_env()
            os.environ["LOCAL_RANK"] = str(args.local_rank)
            os.environ["RANK"] = str(args.rank)
            os.environ["WORLD_SIZE"] = str(args.world_size)
            dist.init_process_group(
                backend=args.dist_backend,
                init_method=args.dist_url,
                world_size=args.world_size,
                rank=args.rank,
            )
        else:
            args.local_rank, _, _ = world_info_from_env()
            dist.init_process_group(
                backend=args.dist_backend, init_method=args.dist_url
            )
            args.world_size = dist.get_world_size()
            args.rank = dist.get_rank()
        args.distributed = True


def assign_learning_rate(param_group, new_lr):
    param_group["lr"] = new_lr


def _warmup_lr(base_lr, warmup_length, step):
    return base_lr * (step + 1) / warmup_length


def cosine_lr(optimizer, base_lrs, warmup_length, steps):
    if not isinstance(base_lrs, list):
        base_lrs = [base_lrs for _ in optimizer.param_groups]
    assert len(base_lrs) == len(optimizer.param_groups)

    def _lr_adjuster(step):
        for param_group, base_lr in zip(optimizer.param_groups, base_lrs):
            if step < warmup_length:
                lr = _warmup_lr(base_lr, warmup_length, step)
            else:
                e = step - warmup_length
                es = steps - warmup_length
                lr = 0.5 * (1 + np.cos(np.pi * e / es)) * base_lr
            assign_learning_rate(param_group, lr)

    return _lr_adjuster


def pickle_save(classifier, save_path):
    with open(save_path, "wb") as f:
        pickle.dump(classifier.cpu(), f)


def pickle_load(save_path, device=None):
    with open(save_path, "rb") as f:
        classifier = pickle.load(f)
    if device is not None:
        classifier = classifier.to(device)
    return classifier


def instantiate(config, *args, **kwargs):
    if isinstance(config, dict):
        cls = config.pop("_target_", None)
        if cls is None:
            raise ValueError("'_target_' key is required in the config dictionary")

        if isinstance(cls, str):
            module_name, class_name = cls.rsplit(".", 1)
            module = __import__(module_name, fromlist=[class_name])
            cls = getattr(module, class_name)

        for key, value in config.items():
            if isinstance(value, dict) and "_target_" in value:
                config[key] = instantiate(value)

        merged_kwargs = {**config, **kwargs}

        return cls(*args, **merged_kwargs)
    else:
        return config
