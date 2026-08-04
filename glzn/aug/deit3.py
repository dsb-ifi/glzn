from collections.abc import Callable, Sequence
from dataclasses import dataclass

import torch
from PIL import ImageOps
from torch import Tensor
from torch.utils.data import default_collate
from torchvision.transforms import v2
from torchvision.transforms.functional import InterpolationMode

SampleTransform = Callable[[object], object]
CollateTransform = Callable[[Sequence[object]], object]

IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)


@dataclass(frozen=True)
class AugmentationPipeline:
    sample: Callable
    collate: Callable


class Solarization:
    """Solarize a PIL image using the DeiT 3-Augment default threshold."""

    def __init__(self, threshold: int = 128):
        self.threshold = threshold

    def __call__(self, img):
        return ImageOps.solarize(img, threshold=self.threshold)


class DefaultCollateWrapper:
    """Apply a batch transform after ordinary PyTorch collation."""

    def __init__(self, transform: Callable[[Tensor, Tensor], tuple[Tensor, Tensor]]):
        self.transform = transform

    def __call__(self, batch: Sequence[object]) -> tuple[Tensor, Tensor]:
        images, targets = default_collate(list(batch))
        return self.transform(images, targets)


class _MaybeApplyBatchTransform:
    def __init__(
        self,
        transform: Callable[[Tensor, Tensor], tuple[Tensor, Tensor]],
        apply_prob: float,
    ):
        self.transform = transform
        self.apply_prob = apply_prob

    def __call__(self, images: Tensor, targets: Tensor) -> tuple[Tensor, Tensor]:
        if torch.rand(()) >= self.apply_prob:
            return images, targets
        return self.transform(images, targets)


def _gaussian_kernel_size(size: int) -> int:
    kernel_size = max(3, int(round(size * 0.1)))
    return kernel_size if kernel_size % 2 == 1 else kernel_size + 1


def build_deit3_augment(
    size: int = 224,
    *,
    color_jitter: float = 0.3,
    use_src: bool = False,
) -> v2.Compose:
    """Build the DeiT III supervised 3-Augment sample transform.

    The official DeiT repository uses timm's
    ``RandomResizedCropAndInterpolation``. GLZN uses torchvision's
    ``RandomResizedCrop`` with the same output size, scale range, and bicubic
    interpolation to avoid a timm dependency. It also uses torchvision v2's
    native Gaussian blur with sigma ``(0.1, 2.0)`` as an approximation to the
    official PIL radius-based blur, not a pixel-identical implementation.
    """

    if size < 1:
        raise ValueError(f"size must be positive, got {size}.")
    if color_jitter < 0:
        raise ValueError(f"color_jitter must be non-negative, got {color_jitter}.")

    steps: list[Callable] = []
    if use_src:
        steps.extend(
            [
                v2.Resize(size, interpolation=InterpolationMode.BICUBIC),
                v2.RandomCrop(size, padding=4, padding_mode="reflect"),
            ]
        )
    else:
        steps.append(
            v2.RandomResizedCrop(
                size,
                scale=(0.08, 1.0),
                interpolation=InterpolationMode.BICUBIC,
            )
        )
    steps.extend(
        [
            v2.RandomHorizontalFlip(p=0.5),
            v2.RandomChoice(
                [
                    v2.Grayscale(num_output_channels=3),
                    Solarization(),
                    v2.GaussianBlur(
                        kernel_size=_gaussian_kernel_size(size),
                        sigma=(0.1, 2.0),
                    ),
                ]
            ),
        ]
    )
    if color_jitter > 0:
        steps.append(
            v2.ColorJitter(
                brightness=color_jitter,
                contrast=color_jitter,
                saturation=color_jitter,
                hue=0,
            )
    )
    steps.extend(
        [
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(
                mean=IMAGENET_DEFAULT_MEAN,
                std=IMAGENET_DEFAULT_STD,
            ),
        ]
    )
    return v2.Compose(steps)


def build_imagenet_eval_augment(
    size: int = 224,
    *,
    crop_ratio: float = 0.875,
) -> v2.Compose:
    if size < 1:
        raise ValueError(f"size must be positive, got {size}.")
    if not 0 < crop_ratio <= 1:
        raise ValueError(f"crop_ratio must lie in (0, 1], got {crop_ratio}.")
    resize_size = int(size / crop_ratio)
    return v2.Compose(
        [
            v2.Resize(resize_size, interpolation=InterpolationMode.BICUBIC),
            v2.CenterCrop(size),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(
                mean=IMAGENET_DEFAULT_MEAN,
                std=IMAGENET_DEFAULT_STD,
            ),
        ]
    )


def build_mixup_cutmix_collate(
    *,
    num_classes: int | None = None,
    mixup_alpha: float = 0.0,
    cutmix_alpha: float = 0.0,
    apply_prob: float = 1.0,
) -> Callable:
    if mixup_alpha < 0:
        raise ValueError(f"mixup_alpha must be non-negative, got {mixup_alpha}.")
    if cutmix_alpha < 0:
        raise ValueError(f"cutmix_alpha must be non-negative, got {cutmix_alpha}.")
    if not 0 <= apply_prob <= 1:
        raise ValueError(f"apply_prob must lie in [0, 1], got {apply_prob}.")

    enabled = mixup_alpha > 0 or cutmix_alpha > 0
    if not enabled:
        return default_collate
    if num_classes is None or num_classes < 1:
        raise ValueError("num_classes >= 1 is required when MixUp/CutMix is enabled.")

    choices: list[Callable[[Tensor, Tensor], tuple[Tensor, Tensor]]] = []
    if mixup_alpha > 0:
        choices.append(v2.MixUp(num_classes=num_classes, alpha=mixup_alpha))
    if cutmix_alpha > 0:
        choices.append(v2.CutMix(num_classes=num_classes, alpha=cutmix_alpha))
    batch_transform = choices[0] if len(choices) == 1 else v2.RandomChoice(choices)
    if apply_prob < 1.0:
        batch_transform = _MaybeApplyBatchTransform(batch_transform, apply_prob)
    return DefaultCollateWrapper(batch_transform)


def build_deit3_pipeline(
    *,
    size: int = 224,
    color_jitter: float = 0.3,
    use_src: bool = False,
    num_classes: int | None = None,
    mixup_alpha: float = 0.0,
    cutmix_alpha: float = 0.0,
    mixup_cutmix_prob: float = 1.0,
) -> AugmentationPipeline:
    return AugmentationPipeline(
        sample=build_deit3_augment(
            size=size,
            color_jitter=color_jitter,
            use_src=use_src,
        ),
        collate=build_mixup_cutmix_collate(
            num_classes=num_classes,
            mixup_alpha=mixup_alpha,
            cutmix_alpha=cutmix_alpha,
            apply_prob=mixup_cutmix_prob,
        ),
    )
