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

`glzn` is designed to stay minimal and efficient for HPC resources, and minimalisim is what drives the development.

---

## Installation

`pip install git+https://github.com/dsbifi/glzn.git`

`pip install git+ssh://git@github.com/dsbifi/glzn.git`

---

