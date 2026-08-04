# glzn.aug

`glzn.aug` keeps augmentation split by execution site:

```text
sample transforms:
    spatial and photometric augmentation before collation

collate transforms:
    batch-level mixing and target conversion after default collation

training step:
    objective semantics

Processor:
    optimization mechanics
```

The DeiT III sample recipe follows the 3-Augment structure: crop/resize,
horizontal flip, exactly one of grayscale, solarization, or Gaussian blur,
optional color jitter, tensor conversion, and ImageNet normalization. It
intentionally does not use Quix's saturation-for-grayscale approximation.

The official DeiT repository uses timm's `RandomResizedCropAndInterpolation`.
GLZN uses torchvision's `RandomResizedCrop` with the same output size, scale
range, and bicubic interpolation to avoid a timm dependency.

The blur branch uses torchvision v2 `GaussianBlur` with sigma `(0.1, 2.0)` and
an odd kernel size around 10% of the image size. This is the native
torchvision approximation to the official DeiT PIL radius-based blur, not a
pixel-identical implementation.
