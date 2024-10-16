import logging
import math
import random

import webdataset as wds
from webdataset import WebLoader

from transforms import image_transform

logger = logging.getLogger(__name__)

_SHARD_SHUFFLE_SIZE = 2000
_SHARD_SHUFFLE_INITIAL = 500
_SAMPLE_SHUFFLE_SIZE = 5000
_SAMPLE_SHUFFLE_INITIAL = 1000


def get_webdataset(
    cfg, dataset, split, clip_transform, clip_normalization=False, tokenizer=None
) -> WebLoader:

    def _read_wds_info(path, split):
        info = dict()
        with open(f"{path}/classnames.txt", "r") as f:
            info["classnames"] = f.read().splitlines()
        with open(f"{path}/zeroshot_classification_templates.txt", "r") as f:
            info["templates"] = f.read().splitlines()
        with open(f"{path}/{split}/nshards.txt", "r") as f:
            info["num_shards"] = int(f.read().splitlines()[0])
        with open(f"{path}/{split}/nsamples.txt", "r") as f:
            info["num_samples"] = int(f.read().splitlines()[0])
        return info

    is_train = split == "train"
    round_fn = math.floor if is_train else math.ceil
    wds_info = _read_wds_info(dataset["path"], split)

    if cfg["task"]["id"] == "patch":
        transform = image_transform(
            modality=dataset["modality"],
            clip_transform=clip_transform,
            clip_normalization=clip_normalization,
            use_augmentations=is_train,
        )
    elif cfg["task"]["id"] == "align":
        satellite, product_type = dataset["modality"].split("_")
        transform_rgb = image_transform(
            modality=f"RGB_{product_type}",
            clip_transform=clip_transform,
            clip_normalization=clip_normalization,
        )
        # TODO: check S1_RGB use case
        transform_ms = image_transform(
            modality=f"{satellite}_{product_type}",
            clip_transform=clip_transform,
        )
    else:
        raise ValueError(f"Unknown task {cfg['task']}")

    if cfg["is_distributed"] and is_train:
        global_batch_size = cfg["task"][f"{split}_batch_size"] * cfg.world_size
        num_batches = round_fn(wds_info["num_samples"] / global_batch_size)
        num_worker_batches = round_fn(num_batches / cfg.num_workers)
        num_batches = num_worker_batches * cfg.num_workers
        num_samples = num_batches * global_batch_size

        pipeline = [
            wds.ResampledShards(
                "/".join(
                    [
                        dataset["path"],
                        split,
                        f"{{0..{wds_info['num_shards']-1}}}.{'tar.gz' if dataset['compressed'] else 'tar'}",
                    ]
                )
            ),
        ]
    else:
        num_batches = round_fn(
            wds_info["num_samples"] / cfg["task"][f"{split}_batch_size"]
        )
        num_samples = num_batches * cfg["task"][f"{split}_batch_size"]

        pipeline = [
            wds.SimpleShardList(
                "/".join(
                    [
                        dataset["path"],
                        split,
                        f"{{0..{wds_info['num_shards']-1}}}.{'tar.gz' if dataset['compressed'] else 'tar'}",
                    ]
                ),
                seed=cfg["torch"]["seed"],
            ),
        ]

    if is_train and cfg["task"]["shuffle"] and not cfg["is_distributed"]:
        pipeline.append(
            wds.detshuffle(
                bufsize=_SHARD_SHUFFLE_SIZE,
                initial=_SHARD_SHUFFLE_INITIAL,
                seed=cfg["torch"]["seed"],
            ),
        )

    if not cfg["is_distributed"]:
        pipeline.extend(
            [
                wds.split_by_worker,
                wds.tarfile_to_samples(),
            ]
        )
    else:
        pipeline.append(
            wds.tarfile_to_samples(),
        )

    if is_train and cfg["task"]["shuffle"]:
        pipeline.append(
            wds.detshuffle(
                bufsize=_SAMPLE_SHUFFLE_SIZE,
                initial=_SAMPLE_SHUFFLE_INITIAL,
                seed=cfg["torch"]["seed"],
            ),
        )

    if cfg["task"]["id"] == "patch":
        pipeline.extend(
            [
                wds.decode("pil"),
                wds.rename(
                    image=dataset["image_key"],
                    label=dataset["target_key"],
                ),
            ]
        )

        if tokenizer:
            pipeline.append(
                wds.map_dict(
                    image=transform, label=lambda target: tokenizer(target)[0]
                ),
            ),
        else:
            pipeline.extend(
                [
                    wds.map_dict(image=transform),
                    wds.to_tuple("image", "label"),
                ]
            )
    elif cfg["task"]["id"] == "align":
        pipeline.extend(
            [
                wds.decode(),
                wds.rename(
                    image_rgb=dataset["image_key"],
                    image_ms=dataset["image_key"],
                    label=dataset["target_key"],
                ),
                wds.map_dict(
                    image_rgb=transform_rgb,
                    image_ms=transform_ms,
                ),
                wds.to_tuple("image_rgb", "image_ms", "label"),
            ]
        )

    pipeline.append(
        wds.batched(
            (
                cfg["task"][f"{split}_batch_size"] // cfg["torch"]["num_workers"]
                if cfg["torch"]["num_workers"]
                else cfg["task"][f"{split}_batch_size"]
            ),
            partial=not is_train,
        ),
    )

    ds = wds.DataPipeline(*pipeline).with_length(wds_info["num_samples"])

    dataloader = WebLoader(
        dataset=ds,
        batch_size=None,
        shuffle=False,
        num_workers=cfg["torch"]["num_workers"],
        persistent_workers=cfg["torch"]["num_workers"] > 0 if is_train else False,
        pin_memory=True,
    )

    if is_train and cfg["task"]["id"] == "patch":
        dataloader = (
            dataloader.unbatched()
            .shuffle(_SAMPLE_SHUFFLE_SIZE, rng=random.Random(cfg["torch"]["seed"]))
            .batched(cfg["task"][f"{split}_batch_size"])
        )
    else:
        dataloader = dataloader.unbatched().batched(cfg["task"][f"{split}_batch_size"])

    dataloader.with_epoch(num_batches)

    dataloader.name = dataset["id"]
    dataloader.classnames = wds_info["classnames"]
    dataloader.templates = wds_info["templates"]
    dataloader.num_batches = num_batches
    dataloader.num_samples = num_samples

    return dataloader
