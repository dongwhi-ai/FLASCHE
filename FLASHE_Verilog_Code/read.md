# FLASHE: CAM-based AI Accelerator for Pattern Recognition

This repository contains the core Verilog HDL implementation and simulation environment for the **FLASHE** project. FLASHE is a high-efficiency hardware accelerator architecture based on Content-Addressable Memory (CAM), optimized for parallel pattern processing and intelligent feature extraction.

---

## Key Features: End-to-End AI Pipeline

FLASHE implements a complete hardware-based AI pipeline consisting of three major operational phases:

### 1. EdgeCount (Top Filter Selection)
- **Feature Sampling**: Performs hardware-driven frequency analysis on input datasets to identify the most significant 2x2 binary patterns.
- **Top Filter Selection**: Automatically identifies and selects the most representative "Top Filters" to be used as reference kernels, ensuring optimized feature extraction tailored to the target dataset.

### 2. Training Phase
- **On-chip Dictionary Learning**: Supports sequential training by populating the CAM-based dictionary with patterns extracted from the training set.
- **Dynamic Registration**: Efficiently manages pattern registration and class-specific updates within the CAM architecture.

### 3. Scoring & Inference Phase
- **Scoring Module**: A dedicated hardware unit that evaluates the importance of matched patterns based on their ranking and frequency.
- **High-speed Prediction**: Performs real-time classification by aggregating scores from the Scoring Module to generate the final output prediction.

---

## Repository Structure

### 1. `src/` (HDL Source Codes)
This directory contains the core hardware logic. While the architecture consists of multiple sub-modules (such as Matchers, Filters, and Normalizers), they are integrated and managed by the following key files:

- **`TopModule.v`**: The top-level hardware orchestrator. It integrates all sub-components and manages the state transitions for the complete AI pipeline, including **EdgeCount (Top Filter Selection)**, **Training**, and **Scoring-based Inference**.
- **`defs.vh`**: The central configuration hub for the entire system. It defines global constants, image dimensions, quantization thresholds, and the rank-based scoring parameters used across all modules.
- *Note: All supporting sub-modules are instantiated within the TopModule to ensure a unified hardware data-path.*

### 2. `sim/` (Simulation Environment)
Contains the verification environment to validate the hardware logic.
- **`TB_TopModule.v`**: Top-level testbench for full pipeline verification (EdgeCount, Training, and Inference).
- **`bram_log.txt`**: Example output log for internal state tracking and scoring verification.
- *Note: Test datasets (.txt) should be placed here as specified in the paths defined in `defs.vh`.*

---

## Prerequisites

- **Development Tool**: Xilinx Vivado 2024.2 or higher.
- **Target Hardware**: Zynq-7000 SoC (e.g., Zybo Z7) or any FPGA-based development environment.

---

## Instructions for Simulation

1. Create a new Vivado project and add all files from the `src/` directory.
2. Add the testbench file from the `sim/` directory.
3. Configure the `IMAGE_BASE` path in `defs.vh` to match your local simulation data directory (relative paths are recommended).
4. Run **Behavioral Simulation** to observe the EdgeCount selection process followed by the classification performance.

---

## Note on Full Hardware Integration

To protect proprietary hardware design assets and specific integration logic, the following files are **not included** in this public repository:
- Full Vivado Project Files (`.xpr`)
- Block Design (BD) integration scripts (`.tcl`)
- System-level AXI Interconnect and Address Mapping configurations

**If you require the complete hardware integration environment or the scripts for research reproduction or academic review purposes, please contact the corresponding author via email.**

---

## Copyright
Copyright (c) 2026 **Integrated Circuits (IC) Lab, Kyung Hee University**. All rights reserved.
