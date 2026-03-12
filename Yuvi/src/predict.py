import os
import argparse
import json
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
from tqdm import tqdm

class TestDataset(Dataset):
    def __init__(self, ids, image_dir, transform=None):
        self.ids = ids
        self.image_dir = image_dir
        self.transform = transform

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        img_id = str(self.ids[idx])
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
            
        return image, img_id

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, required=True, help='Path to test images folder')
    parser.add_argument('--model_path', type=str, required=True, help='Path to model .pth file')
    parser.add_argument('--output', type=str, required=True, help='Output submission.csv path')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device for inference: {device}")
    
    meta_path = os.path.join(os.path.dirname(args.model_path), 'metadata.json')
    with open(meta_path, 'r') as f:
        metadata = json.load(f)
        
    train_class_names = metadata['class_names']
    f_j = np.array(metadata['class_frequencies'])
    best_power = metadata.get('best_power', 0.5)
    print(f"Using optimized smoothing power: {best_power}")
    
    img_dir = args.data_path
    if os.path.isdir(os.path.join(img_dir, 'images')):
        img_dir = os.path.join(img_dir, 'images')

    parent_dir = os.path.dirname(args.data_path)
    test_csv_path = os.path.join(parent_dir, 'test.csv')
    sample_sub_path = os.path.join(parent_dir, 'sample_submission.csv')

    if os.path.exists(test_csv_path):
        df_test = pd.read_csv(test_csv_path)
        id_col = 'image_id' if 'image_id' in df_test.columns else ('id' if 'id' in df_test.columns else df_test.columns[0])
        test_ids = df_test[id_col].values
    elif os.path.exists(sample_sub_path):
        df_test = pd.read_csv(sample_sub_path)
        test_ids = df_test['id'].values
    else:
        test_ids = [f for f in os.listdir(img_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    if os.path.exists(sample_sub_path):
        df_sample = pd.read_csv(sample_sub_path)
        final_class_order = list(df_sample.columns[1:])
    else:
        final_class_order = train_class_names

    model = models.efficientnet_v2_s(weights=None)
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.4, inplace=True),
        nn.Linear(in_features, len(train_class_names))
    )
    
    # Load state dict FIRST while it is still a normal model
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model = model.to(device)
    
    # --- MULTI-GPU INFERENCE ---
    if torch.cuda.device_count() > 1:
        print(f"Let's use {torch.cuda.device_count()} GPUs for Inference!")
        model = nn.DataParallel(model)
        
    model.eval()

    transform = transforms.Compose([
        transforms.Resize(400),
        transforms.CenterCrop(384),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # Safely doubled batch size for dual GPUs
    test_loader = DataLoader(TestDataset(test_ids, img_dir, transform), batch_size=64, shuffle=False, num_workers=4)

    results = []

    with torch.no_grad():
        for images, img_ids in tqdm(test_loader, desc="Predicting"):
            images = images.to(device)
            logits = model(images)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            
            expected_scores = (7.0 * probs - 1.0) / (f_j ** best_power)
            best_classes = np.argmax(expected_scores, axis=1)
            
            for i in range(len(img_ids)):
                predicted_class_name = train_class_names[best_classes[i]]
                row_dict = {cls: 0 for cls in final_class_order}
                row_dict[predicted_class_name] = 1
                
                original_id = img_ids[i]
                row = [original_id] + [row_dict[cls] for cls in final_class_order]
                results.append(row)

    columns = ['id'] + final_class_order
    df_sub = pd.DataFrame(results, columns=columns)
    df_sub.to_csv(args.output, index=False)
    print(f"Predictions successfully saved to {args.output}")

if __name__ == '__main__':
    main()
