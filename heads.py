import logging
from pathlib import Path

import torch
from tqdm import tqdm

from models import ClassificationHead

logger = logging.getLogger(__name__)


def build_classification_head(model, tokenizer, dataloader, device):
    templates = dataloader.templates
    classnames = dataloader.classnames
    logit_scale = model.logit_scale

    model.eval()
    model.to(device)

    print("Building classification head.")
    with torch.no_grad():
        zeroshot_weights = []
        for classname in tqdm(classnames):
            texts = []
            for t in templates:
                texts.append(t.format(c=classname))
            texts = tokenizer(texts).to(device)
            embeddings = model.encode_text(texts)
            embeddings /= embeddings.norm(dim=-1, keepdim=True)

            embeddings = embeddings.mean(dim=0, keepdim=True)
            embeddings /= embeddings.norm()

            zeroshot_weights.append(embeddings)

        zeroshot_weights = torch.stack(zeroshot_weights, dim=0).to(device)
        zeroshot_weights = torch.transpose(zeroshot_weights, 0, 2)

        zeroshot_weights *= logit_scale.exp()

        zeroshot_weights = zeroshot_weights.squeeze().float()
        zeroshot_weights = torch.transpose(zeroshot_weights, 0, 1)

    classification_head = ClassificationHead(normalize=True, weights=zeroshot_weights)

    return classification_head


def get_classification_head(cfg, model, tokenizer, dataloader):
    filename = (
        Path(cfg["env"]["work_dir"])
        / "classification_heads"
        / f"{cfg['model']['name']}_{cfg['model']['pretrained']}_{dataloader.name}_head.pt"
    )
    filename.parent.mkdir(parents=True, exist_ok=True)
    if filename.exists():
        logger.info(
            f"Classification head for {cfg['model']['name']}_{cfg['model']['pretrained']}_{dataloader.name} found at {filename.as_posix()}"
        )
        return ClassificationHead.load(filename.as_posix()).to(cfg["torch"]["device"])
    logger.info(
        f"Did not find classification head for {cfg['model']['name']}_{cfg['model']['pretrained']}_{dataloader.name} at {filename.as_posix()}, building one from scratch."
    )

    classification_head = build_classification_head(
        model, tokenizer, dataloader, cfg["torch"]["device"]
    )
    classification_head.save(filename.as_posix())
    return classification_head
