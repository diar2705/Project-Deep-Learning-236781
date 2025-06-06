# Deep Learning (CS-236781)

This repository contains A deep learning project for the [CS236781](https://vistalab-technion.github.io/cs236781/) Deep Learning course at the Technion.

It is created by [Diar Batheesh](https://github.com/diar2705) and [Hadi Hboos](https://github.com/HadiHboos1).

---

## 📑 Table of Contents

1. [Project Overview](#-project-overview)  
2. [Repository Structure](#-repository-structure)  
3. [Installation](#-installation)  
4. [Usage](#-usage)  
5. [Parts](#-parts)  
   5.1. [Self-Supervised Autoencoding](#-part-121-self-supervised-autoencoding)  
   5.2. [Classification-Guided Encoding](#-part-122-classification-guided-encoding)  
   5.3. [Structured Latent Spaces](#-part-123-structured-latent-spaces)  
6. [Visualization & Analysis](#-visualization--analysis)  
7. [Results](#-results)  
8. [Hyperparameters](#-hyperparameters)

---


## 🧠 Project Overview

A comprehensive three-part study on representation learning using MNIST and CIFAR-10 datasets, implemented with PyTorch. This project delves into self-supervised autoencoding, classification-guided encoding, and structured latent spaces, showcasing advanced techniques in deep learning.

## Parts:
1. **Self-Supervised Autoencoding** (1.2.1)  
2. **Classification-Guided Encoding** (1.2.2)  
3. **Structured Latent Spaces** (1.2.3, SimCLR)

---

## 📁 Repository Structure

```
.
├── main.py                 # Entry point for all experiments
├── environment.yml         # Conda environment specification
├── utils.py                # t-SNE plotting utilities
├── interpolation.py        # Latent-space interpolation scripts
├── project/
│   ├── trainer.py          # Training loops for all parts
│   ├── models.py           # High-level API (Autoencoder, Encoder and classifier, ...)
│   ├── mnist.py            # MNIST-specific encoder/decoder/classifier
│   └── cifar10.py          # CIFAR-10-specific encoder/decoder/classifier
└── plots/                  # Generated visualizations (t-SNE, loss , accuracy)
```

---

## 🚀 Installation
[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue.svg)]()
[![PyTorch](https://img.shields.io/badge/pytorch-%3E%3D1.7-orange.svg)]()

1. Clone:
   ```
   git clone https://github.com/diar2705/Project-Deep-Learning-236781.git
   cd Project-Deep-Learning-236781
   ```
2. Create Conda env:
   ```
   conda env create -f environment.yml
   conda activate aes
   ```
3. (Optional) OR install via pip:
   ```
   pip install -r requirements.txt
   ```

---

## 🛠️ Usage

Launch any part with:
```
python main.py \
  --data-path /path/to/data \
  --batch-size 64 \
  --latent-dim 128 \
  --device cuda \
  [--mnist] \
  --part {1,2,3}
```
- `--mnist`: run on MNIST (omit for CIFAR-10)  
- `--part 1`: Self-Supervised Autoencoding  
- `--part 2`: Classification-Guided Encoding  
- `--part 3`: Structured Latent Spaces (Contrastive)

---

## 🔍 Part 1 (1.2.1) Self-Supervised Autoencoding

**Objective**  
Learn a compact latent representation via reconstruction.

**Workflow**  
1. Train an Autoencoder (L1 loss).  
2. Freeze encoder, attach a classifier head, fine-tune on labels.

**Key Files**  
- `project/trainer.py`: `AutoencoderTrainer`  
- `project/models.py`: `Autoencoder`, `Part1()`

**Artifacts**  
- Reconstructions → `reconstructed_images/`  
- Loss/accuracy plots → `plots/.../part_1.2.1/`

---

## 🔍 Part 2 (1.2.2) Classification-Guided Encoding

**Objective**  
Jointly learn encoding and classification end-to-end.

**Workflow**  
1. Initialize encoder + classifier.  
2. Train on labels with cross-entropy loss.

**Key Files**  
- `project/trainer.py`: `EnclassifierTrainer`  
- `project/models.py`: `run_model()` (part 2 branch)

**Artifacts**  
- Accuracy plots → `plots/.../part_1.2.2/`  
- t-SNE visualizations of latent space

---

## 🔍 Part 3 (1.2.3) Structured Latent Spaces

**Objective**  
Structure latent space via contrastive learning (NT-Xent, SimCLR).

**Workflow**  
1. Train encoder with `CLRTrainer` using paired augmentations.  
2. Freeze encoder, attach classifier, fine-tune.

**Key Files**  
- `project/trainer.py`: `CLRTrainer`, `NTXentLoss`  
- `project/models.py`: `Part3()`, `run_model()` (part 3 branch)

**Artifacts**  
- Contrastive training curves → `plots/.../part_1.2.3/`  
- Structured t-SNE embeddings

---

## 📈 Visualization & Analysis

- **t-SNE**: latent vs. image space via `utils.plot_tsne()`.  
- **Reconstruction & Interpolation for MNIST**: run `interpolation.py` .  
- All figures saved under `plots/`.

---

## 📜 Results

| Part      | Dataset  | Test acc |
|-----------|----------|----------|
| Part 1    | MNIST    | ~96%     |
|           | CIFAR-10 | ~63%     |
| Part 2    | MNIST    | ~99%     |
|           | CIFAR-10 | ~85%     |
| Part 3    | MNIST    | ~97%     |
|           | CIFAR-10 | ~72%     |

WE NEED TO REPLACE WITH THE REAL RESULTS

---

## ⚙️ Hyperparameters

| Param             | Value         |
|-------------------|---------------|
| batch-size        | 64            |
| latent-dim        | 128           |
| AE learning rate  | 1e-3          |
| CLF learning rate | 2e-3 (part 2) |
| CLR learning rate | 1e-3 (part 3) |
| weight decay      | 1e-4          

---

