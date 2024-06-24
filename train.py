"""
                            training script (both temporal learnign / fine tuning )
Author: Divyanshu Tak

Description:
    Training script for temporal learning and finetuning 

"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data.dataset import random_split
import monai
from dataset2 import MedicalImageDatasetBalancedIntensity3D, TransformationMedicalImageDatasetBalancedIntensity3D
from model import dummy_BackboneNetV2, Classifier, MedicalTransformerLSTM, InverseMergingNetworkWithClassification
import yaml
import wandb
import os 
from tqdm import tqdm 
from torch.optim import Adam
from sklearn.metrics import roc_auc_score
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report
import torch.nn.functional as F
import sys
from torch.optim.lr_scheduler import CyclicLR, CosineAnnealingWarmRestarts
import random 
import numpy as np
from torch.cuda.amp import GradScaler, autocast



# set random seed 
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

# config
with open('config.yml', 'r') as file:
    config = yaml.safe_load(file)

# set gpu
os.environ['CUDA_VISIBLE_DEVICES'] = config["gpu"]["visible_device"]
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)


# set wandb
wandb.init(project=config['logger']['project_name'], name=config['logger']['run_name'], config=config)
torch.set_float32_matmul_precision("medium")


# colate function 
def custom_collate(batch):
    """Handles variable size of the scans and pads the sequence dimension to keep the input sequence length constant acorss subjects."""
    images = [item['image'] for item in batch]
    labels = [item['label'] for item in batch]
    
    # Determine the maximum sequence length 
    max_len = config['data']['collate']   
    padded_images = []
    
    for img in images:
        pad_size = max_len - img.shape[0]
        if pad_size > 0:
            padding = torch.zeros((pad_size,) + img.shape[1:])
            img_padded = torch.cat([img, padding], dim=0)
            padded_images.append(img_padded)
        else:
            padded_images.append(img)

    return {"image": torch.stack(padded_images, dim=0), "label": torch.stack(labels)}


# Datasets
train_dataset = TransformationMedicalImageDatasetBalancedIntensity3D(csv_path=config['data']['csv_file'], root_dir= config["data"]["root_dir"])
val_dataset = MedicalImageDatasetBalancedIntensity3D(csv_path=config['data']['val_csv'], root_dir= config["data"]["root_dir"])
train_loader = DataLoader(train_dataset, batch_size=config["data"]["batch_size"], shuffle=True, collate_fn=custom_collate, num_workers = config["data"]["num_workers"])
val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, collate_fn=custom_collate, num_workers = 1)
  

# load the model 
backbonenet = dummy_BackboneNetV2()
mergingnetwork = InverseMergingNetworkWithClassification()
classifier = Classifier(d_model=2048)
model = MedicalTransformerLSTM(backbonenet, mergingnetwork, classifier) #MedicalTransformerLSTM
print("model loaded!")
model = model.to(device)

if config["train"]["finetune"] == "yes":
    pretrained_dict = torch.load(config["train"]["weights"])
    model.load_state_dict(pretrained_dict["model_state_dict"])
    model.classifier = classifier
    print("fientune checkpoint loaded")


criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=config['optim']['lr'], weight_decay=config["optim"]["weight_decay"])
scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=100, T_mult=1)

best_val_loss = 10000
max_epochs = config['model']['max_epochs']
best_auc_roc = 0
best_auc_pr = 0
best_f1 = -100

# training loop
scaler = GradScaler()
for epoch in range(max_epochs):
    model.train()
    train_loss = 0.0
    train_corrects = 0

    for sample in tqdm(train_loader, desc=f"Training Epoch {epoch}/{max_epochs-1}"):
        inputs = sample['image']
        labels = sample['label'].float()  
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        with autocast():
            outputs = model(inputs)
            loss = criterion(outputs, labels.unsqueeze(1))
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        train_loss += loss.item() * inputs.size(0)
        preds = outputs.sigmoid().round()
        train_corrects += torch.sum(preds == labels.unsqueeze(1).data)

    train_loss = train_loss / len(train_loader.dataset)
    train_acc = train_corrects.double().item() / len(train_loader.dataset)
    wandb.log({"Train Loss": train_loss, "Train Acc": train_acc})

    model.eval()
    val_loss = 0.0
    val_corrects = 0
    all_labels = []
    all_preds = []
    all_probs = []

    with torch.no_grad():
        for sample in tqdm(val_loader, desc=f"Validation Epoch {epoch}/{max_epochs-1}"):
            inputs = sample['image']
            labels = sample['label'].float()
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels.unsqueeze(1))

            val_loss += loss.item() * inputs.size(0)
            probs = outputs.sigmoid()
            preds = probs.round()
            all_labels.extend(labels.cpu().numpy().flatten())
            all_preds.extend(preds.cpu().numpy().flatten())
            all_probs.extend(probs.cpu().numpy().flatten())
            val_corrects += torch.sum(preds == labels.unsqueeze(1).data)

    val_loss = val_loss / len(val_loader.dataset)
    val_acc = val_corrects.double().item() / len(val_loader.dataset)
    auc_roc = roc_auc_score(all_labels, all_probs)
    report = classification_report(all_labels, all_preds)
    f1_scores = f1_score(all_labels, all_preds, average='macro')
    
    # log metrics 
    wandb.log({"Val Loss": val_loss, "Val Acc": val_acc, "AUC ROC": auc_roc, "F1 Score": f1_scores})
    scheduler.step(val_loss)
    print(f"Epoch {epoch}/{max_epochs - 1} Train Loss: {train_loss:.4f} Train Acc: {train_acc:.4f} Val Loss: {val_loss:.4f} Val Acc: {val_acc:.4f} AUC ROC: {auc_roc:.4f}")
    print(report)

    # Check if the current F1 score is the best, if so, save the model
    if f1_scores > best_f1:
        print(f"Improved F1 from {best_f1:.4f} to {f1_scores:.4f}. Saving checkpoint...")
        best_f1 = f1_scores
        checkpoint = {
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': val_loss,
            'epoch': epoch,
            'f1_score': f1_scores
        }
        torch.save(checkpoint, os.path.join(config['logger']['save_dir'], config['logger']['save_name'].format(epoch=epoch, f1=best_f1)))

wandb.finish()