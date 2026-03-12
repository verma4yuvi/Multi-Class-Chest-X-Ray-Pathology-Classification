# Multi-Class Chest X-Ray Pathology Classification

This project implements a deep learning framework for the automated detection of thoracic diseases from chest X-ray images. [cite_start]It utilizes a two-stage pipeline combining **Contrastive Self-Supervised Learning (SimCLR)** and **Supervised Fine-Tuning** to achieve high diagnostic accuracy even with imbalanced medical datasets[cite: 13, 27].

## 📌 Project Overview

The framework is designed to address key challenges in medical imaging, such as limited labeled data and severe class imbalance[cite: 24].

- **Stage 1 (Pre-training):** Uses the SimCLR protocol to train an EfficientNet-v2 backbone on unlabeled images to learn robust visual representations[cite: 14, 55].
- **Stage 2 (Fine-tuning):** Adapts the pretrained backbone using Focal Loss to prioritize rare pathologies and difficult cases[cite: 15, 95].

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
🚀 Usage
1. Training (Phase 1 & 2)
Run the train.py script to perform both contrastive pre-training and supervised fine-tuning.

Bash

python train.py --data_path ./data/ --checkpoint_path ./models/best_model.pth --pretrain_epochs 30 --epochs 30
--data_path: Directory containing train.csv and an images/ folder.

--pretrain_epochs: Number of epochs for the SimCLR phase.

--epochs: Number of epochs for the fine-tuning phase.

2. Inference
Use the predict.py script to generate predictions on new test data.

Bash

python predict.py --data_path ./data/test_images/ --model_path ./models/best_model_kaggle.pth --output submission.csv
The script automatically loads class names and the optimized Threshold Power from the metadata.json file created during training.
