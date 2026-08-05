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

The DINO recipe builds sample-level multi-crop views. `dinov2` is the canonical
default: two global crops at scale `(0.32, 1.0)`, eight local crops at scale
`(0.05, 0.32)`, DINO color jitter / grayscale, view-specific blur and
solarization, tensor conversion, and ImageNet normalization. `dinov1` keeps the
same photometric stack but defaults to the original crop scales `(0.4, 1.0)`
and `(0.05, 0.4)`.

DINO sample transforms produce either a structured DINOv2-style dict or the
original flat crop list. The collate function only default-collates that
structure. Teacher/student routing remains training-step semantics, not
augmentation or processor semantics.
