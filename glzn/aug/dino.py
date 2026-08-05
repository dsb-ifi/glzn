from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Literal

import torch
from torch import Tensor
from torch.utils.data import default_collate
from torchvision.transforms import v2
from torchvision.transforms.functional import InterpolationMode

from .deit3 import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD, Solarization

DINORecipe = Literal["dinov1", "dinov2"]
DINOOutputMode = Literal["dict", "flat"]


@dataclass(frozen=True)
class DINOAugmentationPipeline:
    """DINO sample and collate transforms kept as separate execution sites."""

    sample: Callable
    collate: Callable


def build_dino_augment(
    *,
    recipe: DINORecipe = "dinov2",
    output: DINOOutputMode | None = None,
    global_size: int = 224,
    local_size: int = 96,
    global_crops_scale: tuple[float, float] | None = None,
    local_crops_scale: tuple[float, float] | None = None,
    local_crops_number: int = 8,
) -> Callable[[object], dict[str, object] | list[Tensor]]:
    """Build a DINO multi-crop sample transform.

    ``recipe="dinov2"`` is the canonical modern default and returns a structured
    dict matching the DINOv2 augmentation contract. ``recipe="dinov1"`` changes
    only the default crop scales and defaults to the original flat list return.

    The photometric stack is shared across the recipes:
    color jitter with probability 0.8, grayscale with probability 0.2, and
    view-specific Gaussian blur / solarization probabilities. Teacher/student
    routing is intentionally left to the training step.
    """

    if recipe not in ("dinov1", "dinov2"):
        raise ValueError(f"recipe must be 'dinov1' or 'dinov2', got {recipe!r}.")
    if output is None:
        output = "flat" if recipe == "dinov1" else "dict"
    if output not in ("dict", "flat"):
        raise ValueError(f"output must be 'dict' or 'flat', got {output!r}.")
    _validate_size("global_size", global_size)
    _validate_size("local_size", local_size)
    if local_crops_number < 0:
        raise ValueError(
            f"local_crops_number must be non-negative, got {local_crops_number}."
        )

    if global_crops_scale is None:
        global_crops_scale = (0.4, 1.0) if recipe == "dinov1" else (0.32, 1.0)
    if local_crops_scale is None:
        local_crops_scale = (0.05, 0.4) if recipe == "dinov1" else (0.05, 0.32)
    _validate_scale("global_crops_scale", global_crops_scale)
    _validate_scale("local_crops_scale", local_crops_scale)

    return DINOAugment(
        output=output,
        global1=_crop_branch(
            size=global_size,
            scale=global_crops_scale,
            blur_prob=1.0,
            solarize_prob=0.0,
        ),
        global2=_crop_branch(
            size=global_size,
            scale=global_crops_scale,
            blur_prob=0.1,
            solarize_prob=0.2,
        ),
        local=_crop_branch(
            size=local_size,
            scale=local_crops_scale,
            blur_prob=0.5,
            solarize_prob=0.0,
        ),
        local_crops_number=local_crops_number,
    )


def build_dino_pipeline(
    *,
    recipe: DINORecipe = "dinov2",
    output: DINOOutputMode | None = None,
    global_size: int = 224,
    local_size: int = 96,
    global_crops_scale: tuple[float, float] | None = None,
    local_crops_scale: tuple[float, float] | None = None,
    local_crops_number: int = 8,
) -> DINOAugmentationPipeline:
    return DINOAugmentationPipeline(
        sample=build_dino_augment(
            recipe=recipe,
            output=output,
            global_size=global_size,
            local_size=local_size,
            global_crops_scale=global_crops_scale,
            local_crops_scale=local_crops_scale,
            local_crops_number=local_crops_number,
        ),
        collate=collate_dino_views,
    )


class DINOAugment:
    def __init__(
        self,
        *,
        output: DINOOutputMode,
        global1: v2.Compose,
        global2: v2.Compose,
        local: v2.Compose,
        local_crops_number: int,
    ):
        self.output = output
        self.global1 = global1
        self.global2 = global2
        self.local = local
        self.local_crops_number = local_crops_number

    def __call__(self, image: object) -> dict[str, object] | list[Tensor]:
        global_crops = [self.global1(image), self.global2(image)]
        local_crops = [self.local(image) for _ in range(self.local_crops_number)]
        if self.output == "flat":
            return [*global_crops, *local_crops]
        return {
            "global_crops": global_crops,
            "global_crops_teacher": global_crops,
            "local_crops": local_crops,
            "offsets": (),
        }


def collate_dino_views(batch: Sequence[object]) -> object:
    """Default-collate DINO dict or flat-list multi-view samples."""

    return default_collate(list(batch))


def _crop_branch(
    *,
    size: int,
    scale: tuple[float, float],
    blur_prob: float,
    solarize_prob: float,
) -> v2.Compose:
    steps: list[Callable] = [
        v2.RandomResizedCrop(
            size,
            scale=scale,
            interpolation=InterpolationMode.BICUBIC,
        ),
        v2.RandomHorizontalFlip(p=0.5),
        *_color_jittering(),
    ]
    if blur_prob > 0:
        steps.append(
            v2.RandomApply(
                [
                    v2.GaussianBlur(
                        kernel_size=9,
                        sigma=(0.1, 2.0),
                    )
                ],
                p=blur_prob,
            )
        )
    if solarize_prob > 0:
        steps.append(v2.RandomApply([Solarization()], p=solarize_prob))
    steps.extend(_final_tensor_steps())
    return v2.Compose(steps)


def _color_jittering() -> list[Callable]:
    return [
        v2.RandomApply(
            [
                v2.ColorJitter(
                    brightness=0.4,
                    contrast=0.4,
                    saturation=0.2,
                    hue=0.1,
                )
            ],
            p=0.8,
        ),
        v2.RandomGrayscale(p=0.2),
    ]


def _final_tensor_steps() -> list[Callable]:
    return [
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(
            mean=IMAGENET_DEFAULT_MEAN,
            std=IMAGENET_DEFAULT_STD,
        ),
    ]


def _validate_size(name: str, value: int) -> None:
    if value < 1:
        raise ValueError(f"{name} must be positive, got {value}.")


def _validate_scale(name: str, value: tuple[float, float]) -> None:
    lo, hi = value
    if not 0 < lo <= hi <= 1:
        raise ValueError(f"{name} must satisfy 0 < lo <= hi <= 1, got {value}.")
