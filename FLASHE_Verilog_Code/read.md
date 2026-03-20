# FLASHE: CAM-based AI Accelerator for In-Memory Computing

This repository contains the core Verilog HDL implementation and simulation environment for the **FLASHE** project. FLASHE is a high-efficiency AI accelerator architecture designed for MRAM-based Process-in-Memory (PIM) systems, optimized for pattern matching and pulse computing tasks.

---

## Key Features

- **Ternary Pattern Matching**: 2x2 hardware matcher with wildcard (don't care) capability for robust feature detection.
- **Conditional Convolution**: Power-optimized MAC (Multiply-Accumulate) units gated by pattern match signals to reduce redundant switching activity.
- **Pipelined Normalization**: Two-stage pipelined `WeightedNormalizer` for real-time pixel rescaling and data alignment.
- **MRAM PIM Integration**: Architected for Multi-Project Wafer (MPW) fabrication using MRAM-based digital PIM.

---

## Repository Structure

The project is organized into two primary directories:

### 1. `src/` (HDL Source Codes)
Contains the synthesizable Verilog modules for the hardware architecture.
- `Matcher22.v`: 2x2 ternary pattern matching unit.
- `EdgeConvFilter.v`: Conditional convolution filter for edge detection.
- `WeightedNormalizer.v`: 2-stage pipelined pixel rescaling unit.
- `defs.vh`: Global hardware parameters and quantization thresholds.

### 2. `sim/` (Simulation Environment)
Contains the verification environment to validate the hardware logic.
- `TB_TopModule.v`: Top-level testbench for full pipeline verification (Sampling, Training, and Inference phases).
- `bram_log.txt`: Example output log for internal state debugging and verification.
- *Note: Test datasets (.txt) should be placed here according to the paths defined in `defs.vh`.*

---

## Prerequisites

- **Development Tool**: Xilinx Vivado 2024.2 or higher.
- **Target Hardware**: Zynq-7000 SoC (e.g., Zybo Z7) or custom MRAM-based Digital PIM Chip.

---

## Instructions for Simulation

1. Create a new Vivado project and add all files from the `src/` directory.
2. Add the testbench file from the `sim/` directory.
3. Configure the `IMAGE_BASE` path in `defs.vh` to match your local simulation data directory (e.g., use relative paths `./sim/data/`).
4. Run **Behavioral Simulation** to verify the 10-class training and inference cycles.

---

## Note on Full Hardware Integration
configurations

To protect proprietary hardware design assets and specific MPW interface logic, the following files are **not included** in this public repository:
- Full Vivado Project Files (`.xpr`)
- Block Design (BD) integration scripts (`.tcl`)
- Zynq-7000 PS-PL AXI Interconnect and Address Mapping 
**If you require the complete hardware integration environment or the TCL scripts for research reproduction/review, please contact the corresponding author via email.**

---

## Copyright
Copyright (c) 2026 **Integrated Circuits (IC) Lab, Kyung Hee University**. All rights reserved.