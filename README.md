# ZAYAN: Disentangled Contrastive Transformer for Tabular Remote Sensing

![Python](https://img.shields.io/badge/python-3.9%2B-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-orange.svg)
![Status](https://img.shields.io/badge/status-research--code-purple.svg)
![License](https://img.shields.io/badge/license-MIT-lightgrey.svg)

> **ZAYAN** is a two-stage model for tabular remote sensing data:
> 1. **ZAYAN-CL** – feature-level contrastive learning with redundancy reduction.  
> 2. **ZAYAN-T** – a Transformer classifier that preserves the learned feature geometry.

This repository contains the core implementation (`zayan.py`) and a demo notebook illustrating how to train and evaluate ZAYAN on remote–sensing style tabular datasets (e.g., land cover).

> _This repo is anonymized for peer review (e.g., ICPR 2026 submission). Please avoid adding identifying information._

---

##  Key Ideas

- **Feature-level contrastive pretraining (ZAYAN-CL)**  
  Learns embeddings for each feature by treating features as “instances” across samples, combining:
  - InfoNCE-style contrastive loss  
  - Redundancy reduction via a Gram-matrix decorrelation term

- **Transformer classifier (ZAYAN-T)**  
  - Uses learned feature embeddings as multiplicative “tokens”  
  - Adds positional embeddings over features  
  - Minimizes both classification loss and an MSE “preserve loss” that keeps Transformer outputs close to the CL embeddings

- **End-to-end pipeline (ZAYAN)**  
  - Step 1: Pretrain `ZAYAN_CL` on training features  
  - Step 2: Train `ZAYAN_T` on labeled data using class-balanced loss  
  - Step 3: Evaluate with accuracy, macro precision/recall/F1

---

##  Repository Structure

```text
.
├── zayan.py               # Main implementation: ZAYAN_CL, ZAYAN_T, ZAYAN orchestrator
├── ZAYAN_Experiment.ipynb # Example notebook for training & diagnostics
├── __init__.py            # Makes this directory importable as a package
└── README.md

