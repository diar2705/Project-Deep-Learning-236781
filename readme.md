# Deep Learning (CS-236781)

[![Grade](https://img.shields.io/badge/Grade-95%2F100-brightgreen.svg)]()

This repository contains a deep learning project for the [CS236781](https://vistalab-technion.github.io/cs236781/) Deep Learning course at the Technion.

It is created by [Diar Batheesh](https://github.com/diar2705) and [Hadi Hboos](https://github.com/HadiHboos1).

---

## 📑 Table of Contents

1. [Project Overview](#-project-overview)  
2. [Repository Structure](#-repository-structure)  
3. [Installation](#-installation)  
4. [Parts](#-parts)  
   4.1. [Self-Supervised Autoencoding](#-part-1-121-self-supervised-autoencoding)  
   4.2. [Classification-Guided Encoding](#-part-2-122-classification-guided-encoding)  
   4.3. [Structured Latent Spaces](#-part-3-123-structured-latent-spaces)  
5. [Usage](#️-usage)  
6. [Latent t-SNE & Analysis](#-latent-t-sne--analysis)  
7. [Results](#-results)  
8. [Hyperparameters](#️-hyperparameters)  
9. [License](#-license)  
10. [Contributors](#-contributors)

---

## 🧠 Project Overview

A comprehensive three-part study on representation learning using MNIST and CIFAR-10 datasets, implemented with PyTorch. This project delves into self-supervised autoencoding, classification-guided encoding, and structured latent spaces, showcasing advanced techniques in deep learning.

**🎯 Final Grade: 95/100**

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

1. Clone the repository:
   ```bash
   git clone https://github.com/diar2705/Project-Deep-Learning-236781.git
   cd Project-Deep-Learning-236781
   ```
2. Create and activate Conda environment:
   ```bash
   conda env create -f environment.yml
   conda activate aes
   ```

---

## 🔧 Parts

### 🔍 Part 1 (1.2.1) Self-Supervised Autoencoding

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

**Images:**

**MNIST Test Results:**
![Part 1 MNIST Results](plots/mnist/part%201.2.1/Acc%20loss%20classifier.png)

**MNIST t-SNE Latent Space:**
![Part 1 MNIST t-SNE](plots/mnist/part%201.2.1/latent_tsne.png)

**MNIST Image Reconstructions:**
<div style="display: flex; gap: 10px;">
   <img src="plots/mnist/part%201.2.1/reconstructed_images/image_1.png" alt="MNIST AE" width="120">
   <img src="plots/mnist/part%201.2.1/reconstructed_images/image_2.png" alt="MNIST AE" width="120">
   <img src="plots/mnist/part%201.2.1/reconstructed_images/image_3.png" alt="MNIST AE" width="120">
   <img src="plots/mnist/part%201.2.1/reconstructed_images/image_4.png" alt="MNIST AE" width="120">
   <img src="plots/mnist/part%201.2.1/reconstructed_images/image_5.png" alt="MNIST AE" width="120">
</div>

**MNIST Interpolation Results:**
![Interpolation](plots/mnist/part%201.2.1/interpolation.png)

**CIFAR-10 Test Results:**
![Part 1 CIFAR-10 Results](plots/cifar10/part%201.2.1/Acc%20loss%20classifier.png)

**CIFAR-10 t-SNE Latent Space:**
![Part 1 CIFAR-10 t-SNE](plots/cifar10/part%201.2.1/latent_tsne.png)

**CIFAR-10 Image Reconstructions:**
![MNIST AE](plots/cifar10/part%201.2.1/latent_tsne.png)


---

### 🔍 Part 2 (1.2.2) Classification-Guided Encoding

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

**Images:**

**MNIST Test Results:**
![Part 2 MNIST Results](plots/mnist/part%201.2.2/Acc%20loss.png)

**MNIST t-SNE Latent Space:**
![Part 2 MNIST t-SNE](plots/mnist/part%201.2.2/latent_tsne.png)

**CIFAR-10 Test Results:**
![Part 2 CIFAR-10 Results](plots/cifar10/part%201.2.2/Acc%20loss%20classifier.png)

**CIFAR-10 t-SNE Latent Space:**
![Part 2 CIFAR-10 t-SNE](plots/cifar10/part%201.2.2/latent_tsne.png)

---

### 🔍 Part 3 (1.2.3) Structured Latent Spaces

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

**Images:**

**MNIST Test Results:**
![Part 3 MNIST Results](plots/mnist/part%201.2.3/accuracy_loss_plot.png)

**MNIST t-SNE Latent Space:**
![Part 3 MNIST t-SNE](plots/mnist/part%201.2.3/latent_tsne.png)


**CIFAR-10 Test Results:**
![Part 3 CIFAR-10 Results](plots/cifar10/part%201.2.3/accuracy_loss_plot%20classifier.png)

**CIFAR-10 t-SNE Latent Space:**
![Part 3 CIFAR-10 t-SNE](plots/cifar10/part%201.2.3/latent_tsne.png)


## 🛠️ Usage

Launch any part with:
```bash
python main.py \
  --data-path /path/to/data \
  --batch-size 64 \
  --latent-dim 128 \
  --device cuda \
  [--mnist] \
  --part {1,2,3}
```

**Arguments:**
- `--mnist`: Run on MNIST dataset (omit for CIFAR-10)  
- `--part 1`: Self-Supervised Autoencoding  
- `--part 2`: Classification-Guided Encoding  
- `--part 3`: Structured Latent Spaces (Contrastive)

---

## 📈 Latent t-SNE & Analysis

- **t-SNE**: latent vs. image space via `utils.plot_tsne()`.  
- **Reconstruction & Interpolation for MNIST**: run `interpolation.py` .  
- All figures saved under `plots/`.

---

## 📜 Results

| Part      | Dataset  | Test Accuracy |
|-----------|----------|---------------|
| Part 1    | MNIST    | ~96%          |
|           | CIFAR-10 | ~63%          |
| Part 2    | MNIST    | ~99%          |
|           | CIFAR-10 | ~85%          |
| Part 3    | MNIST    | ~97%          |
|           | CIFAR-10 | ~72%          |

---

## ⚙️ Hyperparameters

| Parameter         | Value         |
|-------------------|---------------|
| Batch Size        | 64            |
| Latent Dimension  | 128           |
| AE Learning Rate  | 1e-3          |
| CLF Learning Rate | 2e-3 (part 2) |
| CLR Learning Rate | 1e-3 (part 3) |
| Weight Decay      | 1e-4          |

---

## 📄 License

This project is for educational purposes as part of CS-236781 Deep Learning course at Technion.

## 🤝 Contributors

- [Diar Batheesh](https://github.com/diar2705)
- [Hadi Hboos](https://github.com/HadiHboos1)

