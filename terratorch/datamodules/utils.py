# Copyright contributors to the Terratorch project

import re
from collections.abc import Callable, Iterable

import albumentations as A
import numpy as np
import torch
import logging
logger = logging.getLogger("terratorch")


def wrap_in_compose_is_list(transform_list):
    # set check shapes to false because of the multitemporal case
    return A.Compose(transform_list, is_check_shapes=False) if isinstance(transform_list, Iterable) else transform_list


def check_dataset_stackability(dataset, batch_size: int, max_checks: int | None = 100) -> bool:
    if max_checks is None or len(dataset) <= max_checks:
        random_indexes = np.arange(len(dataset))
    else:
        random_indexes = np.random.randint(low=0, high=len(dataset), size=max_checks)
    shapes = np.array([dataset[idx]["image"].shape for idx in random_indexes])

    if len(shapes) > 1:
        if np.array_equal(np.max(shapes, 0), np.min(shapes, 0)):
            return batch_size
        else:
            logger.warning(
                "The batch samples can't be stacked, since they don't have the same dimensions. Setting batch_size=1."
            )
            return 1
    else:
        return batch_size


def check_dataset_stackability_dict(dataset, batch_size: int, max_checks: int | None = 100) -> bool:
    """Check stackability with item['image'] being a dict."""
    if max_checks is None or len(dataset) <= max_checks:
        random_indexes = np.arange(len(dataset))
    else:
        random_indexes = np.random.randint(low=0, high=len(dataset), size=max_checks)

    shapes = {}
    for idx in random_indexes:
        for mod, value in dataset[idx]["image"].items():
            if mod in shapes:
                shapes[mod].append(value.shape)
            else:
                shapes[mod] = [value.shape]

    if all(np.array_equal(np.max(s, 0), np.min(s, 0)) for s in shapes.values()):
        return batch_size
    else:
        logger.warning(
            "The batch samples can't be stacked, since they don't have the same dimensions. Setting batch_size=1."
        )
        return 1


class NormalizeWithTimesteps(Callable):
    def __init__(self, means, stds):
        super().__init__()
        self.means = means  # (C, T)
        self.stds = stds  # (C, T)

    def __call__(self, batch):
        image = batch["image"]

        if len(image.shape) == 5:  # (B, T, C, H, W)
            means = (
                torch.tensor(self.means, device=image.device)
                .transpose(0, 1)
                .reshape(1, image.shape[1], image.shape[2], 1, 1)
            )
            stds = (
                torch.tensor(self.stds, device=image.device)
                .transpose(0, 1)
                .reshape(1, image.shape[1], image.shape[2], 1, 1)
            )

        elif len(image.shape) == 4:  # (B, C, H, W)
            means = torch.tensor(self.means, device=image.device).mean(dim=1).view(1, image.shape[1], 1, 1)
            stds = torch.tensor(self.stds, device=image.device).mean(dim=1).view(1, image.shape[1], 1, 1)

        else:
            msg = f"Expected batch to have 5 or 4 dimensions, but got {len(image.shape)}"
            raise Exception(msg)

        batch["image"] = (image - means) / stds
        return batch
