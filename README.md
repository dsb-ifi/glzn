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

For `glzn.data` usage details (including grouping mode behavior and constraints), see `glzn/data/README.md`.

`glzn` is designed to stay minimal and efficient for HPC resources, and minimalisim is what drives the development.

---

## Installation

`pip install git+https://github.com/dsbifi/glzn.git`

`pip install git+ssh://git@github.com/dsbifi/glzn.git`

---

## TODOs:

- [ ] `data` submodule
    - [X] iTar implementation.
    - [X] Basic grouping support.
    - [X] Stem search and extraction.
    - [X] Improved extension filtering.
    - [X] Low overhead stateful sampling capability.
    - [ ] Add optional encoders.
        - [ ] blosc2-openhtj2k.
        - [ ] pillow-jxl-plugin.
        - [ ] Additional video codecs.
        - [ ] Seismic data support.
    - [ ] Add encoder based grouping format.
    - [ ] Collator factory with support for NamedTuple or dict from Dataloader.
- [X] `cfg` submodule.
    - [X] Pydantic type verification.
    - [X] Presedence logic.
- [X] `parse` module, mostly LLRD support for now.
- [ ] `log` module for rudimentary logging to jsonl and stdout.
    - [X] Basic logging support.
    - [ ] Add Aim support.
- [ ] `proc` submodule
    - [X] Simple `ema` wrapper.
    - [ ] Simple `sched` module.
        - [X] `wrap` module for wrapping scheduled events.
        - [ ] Merge `sched` and `wrap` into one submodule.
    - [ ] Basic `optim` module with commonly used optimizers.
        - [ ] cAdamW, StableAdamW, cStableAdamW
        - [ ] LAMB, cLAMB
    - [X] `step` module, tracks relevant training / validation phases.
        - [X] `StepState` class, for immutables.
        - [X] `StepTelemetry` class (clock for run start, etc.).
        - [X] `StepTracker` class for full experiment tracking.
    - [ ] Main `proc` module for context-based batch processing.
        - [ ] Gradient clipping support
            - [ ] Add logic for adaptively selecting gradient clipping based on optimizer functionality.
        - [ ] Gradient accumulation support, in conj. with `step`.
        - [ ] AMP support / gradient scaling.
        - [ ] Scheduling support through `wrap`.
        - [ ] Simple logging via `log` + `step` modules.
        - [ ] Context manager implementation.
