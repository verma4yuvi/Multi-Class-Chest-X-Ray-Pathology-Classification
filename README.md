# Multi-Class Chest X-Ray Pathology Classification

This project implements a deep learning framework for the automated detection of thoracic diseases from chest X-ray images. [cite_start]It utilizes a two-stage pipeline combining **Contrastive Self-Supervised Learning (SimCLR)** and **Supervised Fine-Tuning** to achieve high diagnostic accuracy even with imbalanced medical datasets[cite: 13, 27].

## 📌 Project Overview

[cite_start]The framework is designed to address key challenges in medical imaging, such as limited labeled data and severe class imbalance[cite: 24].

- [cite_start]**Stage 1 (Pre-training):** Uses the SimCLR protocol to train an EfficientNet-v2 backbone on unlabeled images to learn robust visual representations[cite: 14, 55].
- [cite_start]**Stage 2 (Fine-tuning):** Adapts the pretrained backbone using Focal Loss to prioritize rare pathologies and difficult cases[cite: 15, 95].

## 🛠️ Requirements

The project requires Python 3.8+ and the following libraries:
- `torch >= 2.0.0`
- `torchvision >= 0.15.0`
- `pandas >= 1.5.0`
- `numpy >= 1.23.0`
- `scikit-learn >= 1.2.0`
- `Pillow >= 9.0.0`
- `tqdm >= 4.65.0`

Install all dependencies using:
```bash
pip install -r requirements.txt
