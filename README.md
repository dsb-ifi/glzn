<div align="center">

<img src="rsc/glzn_lightgrey.svg" alt="glzn vix est" width="200"/>

# glzn

</div>

`glzn` (/²ɡlɪsːn/) is a minimal library to facilitate rapid training and inference of ML / AI for research, developed by the Digital Signal Processing and Image Analysis Group at the Institute of Informatics at the University of Oslo.

The name `glzn` is a textese disemvowelment of *glissen*, meaning sparse in Norwegian.

---

## Goal and Scope

`glzn` is packaged in submodules, each designed to facilitate a specific role in running machine learning experiments, primarily with PyTorch.

- `glzn.data`: [WDS](https://github.com/webdataset/webdataset) format data wrapper for handling data of different modalities.
- `glzn.log`: local logging functionality, can be paired with [Aim](https://github.com/aimhubio/aim) for more extensive reporting.
- `glzn.cfg`: config and argparse module to handle experiments with [pydantic](https://docs.pydantic.dev/latest/).
- `glzn.proc`: training and validation processor for runs, providing optimization in a neatly packaged context manager.

Planned submodules:

- `glzn.optim`: Commonly used optimizers for large scale image training.
- `glzn.aug`: Augmentation factories for commonly used setups.
- `glzn.parse`: Run factories for common supervised / self-supervised vision pipelines.


`glzn` is designed to stay minimal and efficient for HPC resources, and minimalisim is what drives the development.

---

## Installation

`pip install git+https://github.com/dsbifi/glzn.git`

`pip install git+ssh://git@github.com/dsbifi/glzn.git`

---

## TODOs:

- [X] `data` submodule for data handling.
    - [X] iTar implementation.
    - [X] Basic grouping support.
    - [X] Stem search and extraction.
    - [X] Improved extension filtering.
    - [X] Low overhead stateful sampling capability.
    - [ ] Add-ons (low priority):
        - [ ] Add optional encoders.
            - [ ] blosc2-openhtj2k.
            - [ ] pillow-jxl-plugin.
            - [ ] Additional video codecs.
            - [ ] Seismic data support.
        - [ ] Add encoder based grouping format.
        - [ ] Collator factory with support for NamedTuple or dict from Dataloader.
- [ ] `aug` submodule for augmentations.
    - [ ] Standard ViT Augmentations.
    - [ ] DEIT3 Augmentations.
    - [ ] DINO / iBOT Augmentations.
        - [ ] DINOv2 / v3 support.
    - [ ] MAE Augmentations.
- [X] `cfg` submodule for config declaration.
    - [X] Pydantic type verification.
    - [X] Presedence logic.
- [ ] `parse` module for modular approach to central config / run parsing.
    - [ ] LLRD parsing support.
    - [ ] Factories for creating runs for supervised training.
        - [ ] IN1k training.
        - [ ] IN22k training.
        - [ ] COCO Segmentation training.
        - [ ] COCO Detection / Instance Seg. training.
    - [ ] Factories for creating runs for self-supervised training.
        - [ ] DINO (no MIM)
        - [ ] iBOT / DINOv2 / DINOv3
        - [ ] MAE / MIMR (MIM Refiner)
- [X] `log` module for rudimentary logging to jsonl and stdout.
    - [X] Basic logging support.
    - [ ] Add-ons (medium priority):
        - [ ] Add Aim support.
        - [ ] W&B support (low priority, locks users into pay-to-use)
- [ ] `optim` module with commonly used optimizers not covered by PyTorch.
    - [ ] cAdamW, StableAdamW, cStableAdamW.
    - [ ] LAMB, cLAMB.
    - [ ] Flags / registry for adaptive selection of gradient clipping (based on optimizer functionality).
    - [ ] Add-ons (medium priority):
        - [ ] Scion.
- [X] `proc` submodule for train / validation processing and wrappers.
    - [X] Simple `ema` wrapper.
    - [X] Simple `sched` module.
        - [X] Meta style precomputed array based schedulers.
        - [X] `wrap` module for wrapping scheduled events.
    - [X] `step` module, tracks relevant training / validation phases.
        - [X] `StepState` class, for immutables.
        - [X] `StepTelemetry` class (clock for run start, etc.).
        - [X] `StepTracker` class for full experiment tracking.
    - [X] Main `proc` module for context-based batch processing.
        - [X] Gradient clipping support
        - [X] Gradient accumulation support, in conj. with `step`.
        - [X] AMP support / gradient scaling.
        - [X] Scheduling support through `wrap`.
        - [X] Simple logging via `log` + `step` modules.
        - [X] Context manager implementation.
