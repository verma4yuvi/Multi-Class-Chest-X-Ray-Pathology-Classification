import os
import argparse
import json
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, f1_score, recall_score, precision_score
from PIL import Image, ImageFilter
from tqdm import tqdm
import random

# ==========================================
# 1. DATASETS (SimCLR & Supervised)
# ==========================================

class SimCLRDataset(Dataset):
    """Dataset for Phase 1: Returns two differently augmented views of the same image."""
    def __init__(self, dataframe, image_dir, transform=None):
        self.dataframe = dataframe
        self.image_dir = image_dir
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        img_id = str(self.dataframe.iloc[idx, 0])
        img_path = os.path.join(self.image_dir, img_id)
        
        if not os.path.exists(img_path):
            if os.path.exists(f"{img_path}.png"): img_path = f"{img_path}.png"
            elif os.path.exists(f"{img_path}.jpg"): img_path = f"{img_path}.jpg"
            elif os.path.exists(f"{img_path}.jpeg"): img_path = f"{img_path}.jpeg"

        if not os.path.exists(img_path):
            raise FileNotFoundError(f"\nCRITICAL ERROR: Cannot find image '{img_id}' anywhere inside {self.image_dir}\n")
            
        image = Image.open(img_path).convert('RGB')
        
        # Apply the exact same transform pipeline twice to get two different random views
        view_1 = self.transform(image)
        view_2 = self.transform(image)
        return view_1, view_2

class XRayDataset(Dataset):
    """Dataset for Phase 2: Returns the image and its one-hot encoded labels."""
    def __init__(self, dataframe, image_dir, transform=None):
        self.dataframe = dataframe
        self.image_dir = image_dir
        self.transform = transform
        self.labels = self.dataframe.iloc[:, 1:].values.astype(np.float32)
        self.targets = np.argmax(self.labels, axis=1)

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        img_id = str(self.dataframe.iloc[idx, 0])
        img_path = os.path.join(self.image_dir, img_id)
        
        if not os.path.exists(img_path):
            if os.path.exists(f"{img_path}.png"): img_path = f"{img_path}.png"
            elif os.path.exists(f"{img_path}.jpg"): img_path = f"{img_path}.jpg"
            elif os.path.exists(f"{img_path}.jpeg"): img_path = f"{img_path}.jpeg"

        if not os.path.exists(img_path):
            image = Image.new('RGB', (384, 384))
        else:
            image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
            
        return image, self.targets[idx]

# ==========================================
# 2. LOSS FUNCTIONS
# ==========================================

class NTXentLoss(nn.Module):
    """Normalized Temperature-scaled Cross Entropy Loss for SimCLR"""
    def __init__(self, temperature=0.5):
        super(NTXentLoss, self).__init__()
        self.temperature = temperature
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, z_i, z_j):
        batch_size = z_i.shape[0]
        z = torch.cat((z_i, z_j), dim=0)
        sim_matrix = torch.matmul(z, z.T) / self.temperature
        sim_matrix.fill_diagonal_(-float('inf'))
        
        labels = torch.cat((torch.arange(batch_size) + batch_size, torch.arange(batch_size)), dim=0).to(z.device)
        return self.criterion(sim_matrix, labels)

class FocalLoss(nn.Module):
    """Focal Loss for handling severe class imbalance in Phase 2"""
    def __init__(self, gamma=2.0):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.ce = nn.CrossEntropyLoss(reduction='none')

    def forward(self, inputs, targets):
        ce_loss = self.ce(inputs, targets)
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        return focal_loss.mean()

def calculate_competition_score(preds_onehot, targets_onehot):
    TP = np.sum((preds_onehot == 1) & (targets_onehot == 1), axis=0)
    FP = np.sum((preds_onehot == 1) & (targets_onehot == 0), axis=0)
    FN = np.sum((preds_onehot == 0) & (targets_onehot == 1), axis=0)
    N_c = np.sum(targets_onehot == 1, axis=0)
    
    N_c_safe = np.where(N_c == 0, 1, N_c)
    score_c = (TP - FP - 5 * FN) / N_c_safe
    score_c = np.where(N_c == 0, 0, score_c) 
    return np.mean(score_c)

# ==========================================
# 3. MODELS
# ==========================================

class SimCLRModel(nn.Module):
    def __init__(self, base_model, out_dim=128):
        super().__init__()
        self.backbone = base_model
        in_features = self.backbone.classifier[1].in_features
        
        # Remove the classification head to output pure feature representations
        self.backbone.classifier = nn.Identity()
        
        # Attach the SimCLR MLP Projection Head
        self.projector = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, out_dim)
        )

    def forward(self, x):
        features = self.backbone(x)
        projections = self.projector(features)
        # L2 Normalize the projections for the cosine similarity math in NT-Xent
        return nn.functional.normalize(projections, dim=1)

# ==========================================
# 4. MAIN PIPELINE
# ==========================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, required=True, help='Path to folder containing train.csv and images/')
    parser.add_argument('--checkpoint_path', type=str, required=True, help='Base path to save best_model.pth')
    parser.add_argument('--pretrain_epochs', type=int, default=30, help='Epochs for Phase 1 SimCLR')
    parser.add_argument('--epochs', type=int, default=30, help='Epochs for Phase 2 Supervised Fine-Tuning')
    parser.add_argument('--batch_size', type=int, default=64) 
    args = parser.parse_args()

    base_dir = os.path.dirname(args.checkpoint_path)
    base_name = os.path.basename(args.checkpoint_path).replace('.pth', '')
    model_loss_path = os.path.join(base_dir, f"{base_name}_loss.pth")
    model_kaggle_path = os.path.join(base_dir, f"{base_name}_kaggle.pth")
    simclr_backbone_path = os.path.join(base_dir, f"{base_name}_simclr_backbone.pth")

    os.makedirs(base_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    csv_path = os.path.join(args.data_path, 'train.csv')
    img_dir = os.path.join(args.data_path, 'images')
    if os.path.isdir(os.path.join(img_dir, 'images')):
        img_dir = os.path.join(img_dir, 'images')
        
    df = pd.read_csv(csv_path)
    class_names = list(df.columns[1:])
    class_counts = df.iloc[:, 1:].sum().values
    class_frequencies = class_counts / class_counts.sum()

    train_df, val_df = train_test_split(df, test_size=0.15, random_state=42, stratify=np.argmax(df.iloc[:, 1:].values, axis=1))

    # --- PHASE 1: SIMCLR PRETRAINING ---
    if args.pretrain_epochs > 0 and not os.path.exists(simclr_backbone_path):
        print("\n" + "="*50)
        print("PHASE 1: CONTRASTIVE SELF-SUPERVISED PRETRAINING (SimCLR)")
        print("="*50)
        
        # Heavy augmentations are critical for SimCLR to work
        simclr_transform = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.5, 1.0)), # Smaller crop for speed during pretraining
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
            transforms.RandomGrayscale(p=0.2),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        # Use full dataset for pretraining (unlabeled)
        simclr_loader = DataLoader(SimCLRDataset(df, img_dir, simclr_transform), batch_size=args.batch_size * 2, shuffle=True, num_workers=4)
        
        base_encoder = models.efficientnet_v2_s(weights=models.EfficientNet_V2_S_Weights.DEFAULT)
        simclr_model = SimCLRModel(base_encoder).to(device)
        
        if torch.cuda.device_count() > 1:
            simclr_model = nn.DataParallel(simclr_model)
            
        criterion_simclr = NTXentLoss(temperature=0.5).to(device)
        optimizer_simclr = optim.AdamW(simclr_model.parameters(), lr=3e-4, weight_decay=1e-4)
        
        for epoch in range(args.pretrain_epochs):
            simclr_model.train()
            total_loss = 0
            for view1, view2 in tqdm(simclr_loader, desc=f"SimCLR Epoch {epoch+1}/{args.pretrain_epochs}"):
                view1, view2 = view1.to(device), view2.to(device)
                optimizer_simclr.zero_grad()
                
                z_i = simclr_model(view1)
                z_j = simclr_model(view2)
                
                loss = criterion_simclr(z_i, z_j)
                loss.backward()
                optimizer_simclr.step()
                total_loss += loss.item()
                
            print(f"SimCLR Epoch {epoch+1} | Contrastive Loss: {total_loss/len(simclr_loader):.4f}")
            
        # Save only the pretrained backbone (discard the MLP projector)
        extracted_backbone = simclr_model.module.backbone if isinstance(simclr_model, nn.DataParallel) else simclr_model.backbone
        torch.save(extracted_backbone.state_dict(), simclr_backbone_path)
        print(f"[*] SimCLR Pretraining Complete. Backbone saved to {simclr_backbone_path}")

    # --- PHASE 2: SUPERVISED FINE-TUNING ---
    print("\n" + "="*50)
    print("PHASE 2: SUPERVISED FINE-TUNING (Focal Loss)")
    print("="*50)

    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(384, scale=(0.75, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    val_transform = transforms.Compose([
        transforms.Resize(400),
        transforms.CenterCrop(384),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    train_loader = DataLoader(XRayDataset(train_df, img_dir, train_transform), batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(XRayDataset(val_df, img_dir, val_transform), batch_size=args.batch_size, shuffle=False, num_workers=4)

    # Initialize model
    model = models.efficientnet_v2_s(weights=None) # We will load our SimCLR weights instead of ImageNet
    
    # Load SimCLR Backbone Weights
    if os.path.exists(simclr_backbone_path):
        model.classifier = nn.Identity() # Temporarily match the SimCLR backbone structure
        model.load_state_dict(torch.load(simclr_backbone_path, map_location=device))
        print("=> Successfully loaded SimCLR Pretrained Backbone!")
    else:
        print("=> Warning: SimCLR backbone not found. Training from scratch.")

    # Reattach the Classification Head for supervised tuning
    in_features = 1280 # Standard out features for EfficientNet-V2-S
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.4, inplace=True), 
        nn.Linear(in_features, len(class_names))
    )
    
    model = model.to(device)
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    criterion = FocalLoss(gamma=2.0).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-2)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=2)

    best_val_loss = float('inf')
    best_val_score = -float('inf')
    meta_path = os.path.join(base_dir, 'metadata.json')

    # Optional: Resume from previous finetuning checkpoint
    if os.path.exists(model_kaggle_path):
        print(f"\n=> Found existing Fine-tuned checkpoint at '{model_kaggle_path}'")
        state_dict = torch.load(model_kaggle_path, map_location=device)
        if isinstance(model, nn.DataParallel):
            model.module.load_state_dict(state_dict)
        else:
            model.load_state_dict(state_dict)

    for epoch in range(args.epochs):
        model.train()
        train_loss = 0.0
        for images, targets in tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Train]"):
            images, targets = images.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * images.size(0)
            
        model.eval()
        val_loss = 0.0
        all_probs = []
        all_targets = []
        
        with torch.no_grad():
            for images, targets in tqdm(val_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Val]"):
                images, targets = images.to(device), targets.to(device)
                outputs = model(images)
                loss = criterion(outputs, targets)
                val_loss += loss.item() * images.size(0)
                
                probs = torch.softmax(outputs, dim=1)
                all_probs.append(probs.cpu().numpy())
                all_targets.append(targets.cpu().numpy())

        train_loss /= len(train_loader.dataset)
        val_loss /= len(val_loader.dataset)
        scheduler.step(val_loss)
        
        all_probs = np.vstack(all_probs)
        all_targets = np.concatenate(all_targets)
        targets_onehot = np.zeros((len(all_targets), len(class_names)))
        targets_onehot[np.arange(len(all_targets)), all_targets] = 1
        
        try:
            auc_scores = roc_auc_score(targets_onehot, all_probs, average=None)
            macro_auc = np.mean(auc_scores)
        except ValueError:
            auc_scores = np.zeros(len(class_names))
            macro_auc = 0.0 
            
        best_epoch_score = -float('inf')
        best_power_for_epoch = 0.0
        best_f1, best_rec, best_prec = 0, 0, 0
        
        print(f"\n--- EPOCH {epoch+1} EVALUATION ---")
        print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        print(f"Macro AUC-ROC: {macro_auc:.4f}")
        auc_str = ", ".join([f"{class_names[i][:4]}: {auc_scores[i]:.2f}" for i in range(len(class_names))])
        print(f"Per-Class AUC: {auc_str}")
        print("\nThreshold Analysis (Power -> Precision | Recall | F1 | Kaggle Score):")
        
        for power in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0]:
            f_j = np.array(class_frequencies)
            expected_scores = (7.0 * all_probs - 1.0) / (f_j ** power)
            preds = np.argmax(expected_scores, axis=1)
            
            preds_onehot = np.zeros((len(preds), len(class_names)))
            preds_onehot[np.arange(len(preds)), preds] = 1
            
            score = calculate_competition_score(preds_onehot, targets_onehot)
            prec = precision_score(all_targets, preds, average='macro', zero_division=0)
            rec = recall_score(all_targets, preds, average='macro', zero_division=0)
            f1 = f1_score(all_targets, preds, average='macro', zero_division=0)
            
            print(f"  Power {power:.1f} -> Prec: {prec:.4f} | Rec: {rec:.4f} | F1: {f1:.4f} | Score: {score:.4f}")
            
            if score > best_epoch_score:
                best_epoch_score = score
                best_power_for_epoch = power
                best_f1, best_rec, best_prec = f1, rec, prec
                
        print(f"\nBest Settings this Epoch -> Power: {best_power_for_epoch} | Kaggle Score: {best_epoch_score:.4f}")

        state_dict_to_save = model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()
        
        metadata = {
            "class_names": class_names,
            "class_frequencies": class_frequencies.tolist(),
            "best_power": best_power_for_epoch
        }
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(state_dict_to_save, model_loss_path)
            print(f"[*] Saved New Best Loss Model to {model_loss_path}")
            
        if best_epoch_score > best_val_score:
            best_val_score = best_epoch_score
            torch.save(state_dict_to_save, model_kaggle_path)
            with open(meta_path, 'w') as f:
                json.dump(metadata, f)
            print(f"[*] Saved New Best Kaggle Score Model to {model_kaggle_path} (Score: {best_val_score:.4f})")
        print("-" * 50)

if __name__ == '__main__':
    main()
