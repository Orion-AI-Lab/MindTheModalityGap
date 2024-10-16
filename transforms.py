import logging
from pprint import pformat

import torch
from torchvision.transforms import (
    ColorJitter,
    Compose,
    GaussianBlur,
    Lambda,
    RandomApply,
    RandomGrayscale,
    RandomHorizontalFlip,
    RandomVerticalFlip,
    Resize,
    RandomResizedCrop,
    ToPILImage,
    ToTensor,
    CenterCrop,
)

logger = logging.getLogger(__name__)


# https://sentinel.esa.int/web/sentinel/user-guides/sentinel-2-msi
def _s2_l1c_to_rgb(image):
    rgb_composite = image[[3, 2, 1]]
    rgb_composite = (rgb_composite / 3558) * 255
    rgb_composite = torch.clamp(rgb_composite, 0, 255)
    rgb_composite = rgb_composite.to(torch.uint8)
    return rgb_composite


# https://sentinel.esa.int/web/sentinel/user-guides/sentinel-2-msi
def _s2_l2a_to_rgb(image):
    rgb_composite = image[[3, 2, 1]]
    rgb_composite = (rgb_composite / 2000) * 255
    rgb_composite = torch.clamp(rgb_composite, 0, 255)
    rgb_composite = rgb_composite.to(torch.uint8)
    return rgb_composite


def image_transform(
    modality, clip_transform, clip_normalization=False, use_augmentations=False
):
    transforms = []

    if modality == "RGB":
        pass
    elif modality == "RGB_L2A":
        transforms.extend(
            [
                _s2_l2a_to_rgb,
                ToPILImage("RGB"),
            ]
        )
    elif modality == "RGB_L1C":
        transforms.extend(
            [
                _s2_l1c_to_rgb,
                ToPILImage("RGB"),
            ]
        )
    elif modality == "S2_L2A":
        transforms.append(
            Lambda(
                lambda img: torch.cat(
                    (
                        img[:10, :, :],
                        torch.zeros(
                            (1, img[0].shape[0], img[0].shape[1]), dtype=torch.float32
                        ),
                        img[10:, :, :],
                    ),
                    dim=0,
                )
            ),
        )
    elif modality == "S2_L1C":
        transforms.append(Lambda(lambda image: image.float()))
    else:
        raise ValueError(f"Unknown modality {modality}")

    transforms.extend(
        [
            transform
            for transform in clip_transform.transforms
            if (
                isinstance(transform, Resize)
                or isinstance(transform, CenterCrop)
                or isinstance(transform, RandomResizedCrop)
            )
        ]
    )

    if use_augmentations:
        transforms.extend(
            [
                RandomHorizontalFlip(p=0.5),
                RandomVerticalFlip(p=0.5),
            ]
        )

        if "RGB" in modality:
            transforms.extend(
                [
                    RandomApply([ColorJitter(0.4, 0.4, 0.4)], p=0.8),
                    RandomGrayscale(p=0.2),
                    RandomApply(
                        [GaussianBlur(kernel_size=(3, 7), sigma=(0.1, 2.0))], p=0.4
                    ),
                ]
            )

    transforms.append(
        Lambda(
            lambda image: (
                image if isinstance(image, torch.Tensor) else ToTensor()(image)
            )
        )
    )
    if not clip_normalization:
        normalization = Lambda(lambda image: torch.clamp(image / 10000.0, 0, 1))
    else:
        normalization = clip_transform.transforms[-1]

    transforms.append(normalization)

    # TODO: Remove this
    logger.info(f"Using {modality} with \n {pformat(transforms)}")
    return Compose(transforms)
