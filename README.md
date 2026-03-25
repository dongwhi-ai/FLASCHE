# FLASCHE

This repository accompanies the manuscript:

**FLASCHE: A Feedforward Location Activated Spike-based Computational-CAM Hardening Edge-processor Using Hierarchical Feature Extraction for Rapid On-Chip Learning and Inference**

FLASCHE is a memory-centric learning and inference framework designed for edge intelligence. The method is based on Content-Addressable Memory (CAM) / Computational CAM (CCAM) and performs learning and inference without backpropagation, floating-point computation, or multiply-accumulate (MAC) operations. Instead, the framework relies on hardware-friendly pattern matching, counting, sorting, and rank-based scoring.

This repository provides the software implementation for algorithm validation and hardware-related materials for FPGA realization. The public version is organized primarily around the software reproduction flow, while the hardware implementation is documented separately in a dedicated hardware subdirectory.

---

## Overview

Conventional SNN training methods often rely on surrogate gradients, which reintroduce high-precision arithmetic and substantial memory/computation overhead. FLASCHE takes a different approach by reformulating learning as **pattern search → pattern matching → counter accumulation → address reordering/ranking**, making the pipeline more suitable for digital in-memory / near-memory hardware.

The key idea of FLASCHE is to combine:

- **hierarchical feature extraction**
- **top-k local pattern selection**
- **CAM/CCAM-based pattern ranking and scoring**
- **fully feedforward learning and inference**

This makes the method particularly relevant to low-power, low-latency, and on-device adaptive intelligence.

---

## Main Features

- **MAC-free and floating-point-free learning and inference**
  - No backpropagation
  - No floating-point weight updates
  - No multiplier-heavy convolutional training pipeline

- **CAM-native pattern learning**
  - Local binary patterns are counted and ranked
  - Frequently observed patterns are selected as top filters
  - Inference is performed by rank-based scoring over matched patterns

- **Software-hardware co-design**
  - Software implementation for algorithm validation
  - Hardware-oriented implementation for FPGA realization
  - Public repository structure designed to support reproducibility

- **Hierarchical feature extraction**
  - Edge-level and higher-order pattern extraction
  - Designed to improve representational power beyond shallow single-stage matching

---

## Reported Results

The manuscript reports the following representative results:

- **Software (MNIST):** 97.97% classification accuracy
- **FPGA (Digits):** 99.33% classification accuracy
- **FPGA core learning time:** 6.9 ms for 1,500 training images
- **FPGA core inference time:** 7.5 µs/image at 50 MHz

---

## Repository Organization

This repository is organized around the software implementation, while hardware-specific materials are placed in a dedicated subdirectory.

```text
FLASCHE
├── README.md
├── LICENSE
├── NOTICE
├── requirements.txt
├── src/
│   ├── cam/
│   ├── data/
│   ├── eval/
│   ├── filter/
│   ├── layer/
│   ├── model/
│   ├── pipeline/
│   └── utils/
├── scripts/
├── configs/
├── data/
└── FLASHE_Verilog_Code/
    ├── README.md
    ├── source/
    └── sim/
```

---

## Reproducibility Scope

This repository is intended to support reproducibility at two levels.

### 1. Software-level reproduction

Software-side validation of the FLASCHE pipeline, including pattern extraction, top-filter selection, counting, ranking, and scoring.

### 2. Hardware-level understanding and verification

Hardware-related materials for the FPGA-oriented realization of FLASCHE, including module-level HDL sources and simulation-oriented files, are documented in `FLASHE_Verilog_Code/README.md`.

Because software validation and hardware realization serve different purposes, they are documented separately.

---

## Dataset Preparation

This repository uses the following datasets:

- **MNIST**: Used for software evaluation. The dataset is already provided in this repository under the `data/` directory, so no separate manual download is required.
- **Digits**: Used for FPGA validation as well as software validation. It is loaded from the `scikit-learn` package at runtime and is not included in this repository as a standalone dataset.

---

## Getting Started

### Software

1. Create a Python environment.
2. Install dependencies from `requirements.txt`.
3. Run one of the following commands.

#### Via helper script

```bash
scripts/run_train_inference --config configs/default.yaml
scripts/run_train_inference --config configs/paper_MNIST_configs.yaml
scripts/run_train_inference --config configs/paper_digits_configs.yaml
```

#### Direct Python invocation

```bash
python -m src.pipeline.train_inference --config configs/default.yaml
python -m src.pipeline.train_inference --config configs/paper_MNIST_configs.yaml
python -m src.pipeline.train_inference --config configs/paper_digits_configs.yaml
```

### Hardware

For hardware-related usage, simulation, and implementation notes, see `FLASHE_Verilog_Code/README.md`.

---

## Intended Use

This repository is intended for:

- academic review
- research reproduction
- method understanding
- software/hardware comparison for CAM-based on-chip learning

It is not presented as a general-purpose deep learning framework. Instead, it is a research repository accompanying the FLASCHE manuscript and focused on reproducibility of the proposed method.

---

## Citation

If you use this repository in academic work, please cite the corresponding paper.

**Manuscript citation**

```bibtex
@article{kim2026flasche,
  title={FLASCHE: A Feedforward Location Activated Spike-based Computational-CAM Hardening Edge-processor Using Hierarchical Feature Extraction for Rapid On-Chip Learning and Inference},
  author={Kim, Dongwhi and Rhim, Hyunki and Choi, Huiseong and Kim, Sumin and Park, Joohwan and Hong, Choong Seon and Hong, Sang Hoon},
  journal={Submissions being processed at Neurocomputing},
  year={2026}
}
```

Please update the citation information after publication.

---

## License

This repository is released under the **Apache License 2.0**. See the [LICENSE](./LICENSE) file for the full license text.

The license applies to the original code and documentation in this repository unless otherwise noted.

---

## Third-Party Data and Components

Datasets, external libraries, and third-party components may be subject to their own licenses or terms of use.

In particular:

- **MNIST** is bundled with this repository.
- **Digits** is accessed through the `scikit-learn` package and is not redistributed here as a standalone dataset.
- Any third-party code, packages, or assets remain subject to their respective licenses.

Users are responsible for complying with the applicable terms of any external data or third-party dependencies used with this repository.
