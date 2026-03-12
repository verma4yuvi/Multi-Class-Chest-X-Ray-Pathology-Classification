# Multi-Class-Chest-X-Ray-Pathology-Classification
In this work, we propose a two-stage training pipeline that combines contrastive representation learning with supervised fine-tuning. The model first learns general visual represen- tations using contrastive learning and is then adapted for pathology classification using supervised training with focal loss.
## Project Architecture 
The pipeline consists of two distinct training stages:Stage 1: Contrastive Pre-training (SimCLR): An EfficientNet-v2 backbone is pretrained to learn general visual representations by maximizing agreement between differently augmented views of the same image.Stage 2: Supervised Fine-Tuning: The pretrained backbone is adapted for multi-class classification using Focal Loss to focus the model on rare and difficult disease categories.
## Key FeaturesBackbone
EfficientNet-v2-s for efficient scaling and strong feature extraction.Imbalance Handling: Implementation of Focal Loss ($\gamma=2.0$) to mitigate the dominance of the "No Finding" class.Inference Optimization: Uses Threshold Power Optimization to balance precision and recall, and Test-Time Augmentation (TTA) for stable predictions.Performance: Achieves a Macro AUC-ROC of 0.8287 across 19 different pathologies.
Gemini said
README: Multi-Class Chest X-Ray Pathology Classification
This project implements a robust deep learning framework for diagnosing thoracic diseases from chest X-ray images. The system addresses common medical imaging challenges, such as limited labeled data and severe class imbalance, by combining Contrastive Self-Supervised Learning (SimCLR) with Supervised Fine-Tuning using Focal Loss.
+4

## Project Architecture
The pipeline consists of two distinct training stages:
Stage 1: Contrastive Pre-training (SimCLR): An EfficientNet-v2 backbone is pretrained to learn general visual representations by maximizing agreement between differently augmented views of the same image.
Stage 2: Supervised Fine-Tuning: The pretrained backbone is adapted for multi-class classification using Focal Loss to focus the model on rare and difficult disease categories.

## Key Features
Backbone: EfficientNet-v2-s for efficient scaling and strong feature extraction.
Imbalance Handling: Implementation of Focal Loss (γ=2.0) to mitigate the dominance of the "No Finding" class.
Inference Optimization: Uses Threshold Power Optimization to balance precision and recall, and Test-Time Augmentation (TTA) for stable predictions.
Performance: Achieves a Macro AUC-ROC of 0.8287 across 19 different pathologies.

## Installation
Ensure you have Python 3.8+ installed. You can install the required dependencies using the provided requirements.txt:
Bash
pip install -r requirements.txt
Core Dependencies:
torch >= 2.0.0 
torchvision >= 0.15.0 
pandas, numpy, scikit-learn, Pillow, tqdm 
