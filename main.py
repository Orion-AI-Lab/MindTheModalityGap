import json
import logging
from datetime import datetime
from pathlib import Path

import open_clip
import torch
import torch.nn as nn

import warnings

from agents import AlignmentAgent, PatchingAgent
from data import get_webdataset
from utils import is_distributed, set_seed

logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore")


def main():
    cfg = json.load(open("conf/config.json"))
    cfg["task"] = json.load(open(f"conf/tasks/{cfg['task']}.json"))

    set_seed(cfg["torch"]["seed"])

    log_dir = (
        Path(cfg["env"]["log_dir"])
        / f"{cfg["model"]["name"]}_{cfg["model"]["pretrained"]}_{cfg["task"]["id"]}"
        / datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
    )
    log_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        filename=log_dir / cfg["env"]["log_file"],
        filemode="w",
        format="%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=logging.INFO,
    )

    (
        clip_model,
        clip_transform_train,
        clip_transform_val,
    ) = open_clip.create_model_and_transforms(
        model_name=cfg["model"]["name"],
        pretrained=cfg["model"]["pretrained"],
        precision=cfg["model"]["precision"],
        device=cfg["model"]["device"],
    )

    if cfg["torch"]["grad_checkpointing"]:
        clip_model.set_grad_checkpointing()
        logger.info("Gradient checkpointing has been enabled.")

    clip_tokenizer = open_clip.get_tokenizer(cfg["model"]["name"])
    model_cfg = open_clip.get_model_config(cfg["model"]["name"])

    cfg["is_distributed"] = is_distributed()
    cfg["embed_dim"] = model_cfg["embed_dim"]
    cfg["image_size"] = (model_cfg["vision_cfg"]["image_size"],) * 2

    # Define the task dataset splits
    task_dataset = dict()
    task_dataset_cfg = json.load(
        open(f"conf/datasets/{cfg['data']['task_dataset']}.json")
    )
    for split in ["train", "test", "val"]:
        task_dataset[split] = get_webdataset(
            cfg,
            dataset=task_dataset_cfg,
            split=split,
            clip_transform=(
                clip_transform_train if split == "train" else clip_transform_val
            ),
            clip_normalization=True,
        )
    cfg["data"]["task_dataset"] = task_dataset_cfg

    # Define the evaluation datasets
    eval_dataset = []
    eval_dataset_cfg = []
    for ds in cfg["data"]["eval_dataset"]:
        ds_cfg = json.load(open(f"conf/datasets/{ds}.json"))
        eval_dataset.append(
            get_webdataset(
                cfg,
                dataset=ds_cfg,
                split="test",
                clip_transform=clip_transform_val,
                clip_normalization=True,
            )
        )
        eval_dataset_cfg.append(ds_cfg)
    cfg["data"]["eval_dataset"] = eval_dataset_cfg

    if cfg["task"]["id"] == "patch":
        agent = PatchingAgent(
            cfg=cfg,
            model=clip_model,
            tokenizer=clip_tokenizer,
            task_dataset=task_dataset,
            eval_dataset=eval_dataset,
        )
    elif cfg["task"]["id"] == "align":
        open_clip.add_model_config(cfg["task"]["student_model"]["model_cfg"])

        modality_encoder = open_clip.create_model(
            model_name=cfg["task"]["student_model"]["model_name"],
            pretrained=cfg["task"]["student_model"]["pretrained"],
            precision=cfg["torch"]["precision"],
            device=cfg["torch"]["device"],
        )

        modality_encoder.visual.trunk.patch_embed.proj = torch.nn.Conv2d(
            13, 384, kernel_size=(16, 16), stride=(16, 16)
        )

        if cfg["torch"]["grad_checkpointing"]:
            modality_encoder.set_grad_checkpointing()
            logger.info("Gradient checkpointing has been enabled for student model.")

        delattr(modality_encoder, "transformer")

        # Load pretrained modality encoder weights (https://github.com/zhu-xlab/SSL4EO-S12)
        msg = modality_encoder.visual.trunk.load_state_dict(
            torch.load(
                "/mnt/hdd/workspaces/clip-satellite/pretrained/alignment/vit_small_patch16_224_sentinel2_all_moco.pth",
                "cpu",
            )["model"],
            strict=False,
        )
        logger.info(f"Modality encoder weights loaded: {msg}")
        modality_encoder.visual.head = nn.Linear(384, cfg["embed_dim"])

        agent = AlignmentAgent(
            cfg=cfg,
            teacher_model=clip_model,
            student_model=modality_encoder,
            tokenizer=clip_tokenizer,
            task_dataset=task_dataset,
            eval_dataset=eval_dataset,
        )
    else:
        raise NotImplementedError

    agent.run()


if __name__ == "__main__":
    main()
