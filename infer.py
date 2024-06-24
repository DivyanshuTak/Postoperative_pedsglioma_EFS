"""
                            Inference script (both temporal learnign / fine tuning )
Author: Divyanshu Tak

Description:
    Inference / testing script. generates a output csv with modeloutput, predictions, ground truth labels

"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset2 import MedicalImageDatasetBalancedIntensity3D
from model import dummy_BackboneNetV2, Classifier, MedicalTransformerLSTM, InverseMergingNetworkWithClassification
import yaml
import os 
from tqdm import tqdm 
from sklearn.metrics import roc_auc_score
from sklearn.metrics import classification_report
import torch.nn.functional as F
import random 
import numpy as np
import pandas as pd 
from sklearn.metrics import confusion_matrix
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix, precision_recall_curve, roc_curve
from sklearn.calibration import calibration_curve
import matplotlib.pyplot as plt 

# set random seed 
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)



# Load config
with open('config.yml', 'r') as file:
    config = yaml.safe_load(file)

# set gpu
os.environ['CUDA_VISIBLE_DEVICES'] = config["gpu"]["visible_device"]
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

torch.set_float32_matmul_precision("medium")


##= 95% CI using bootstrap 
def compute_auc_with_ci(y_true, y_probs, n_bootstraps=1000, ci=0.95):
    """
    Computes the AUC and its 95% Confidence Interval using bootstrapping.

    :param y_true: Ground truth binary labels.
    :param y_probs: Predicted probabilities.
    :param n_bootstraps: Number of bootstrap samples.
    :param ci: Confidence interval level.
    :return: Tuple (AUC, lower bound of CI, upper bound of CI).
    """
    auc = roc_auc_score(y_true, y_probs)
    bootstrapped_scores = []
    
    rng = np.random.default_rng(seed) 
    
    for _ in range(n_bootstraps):
        indices = rng.choice(len(y_true), size=len(y_true), replace=True)
        if len(np.unique(y_true[indices])) < 2:
            continue

        score = roc_auc_score(y_true[indices], y_probs[indices])
        bootstrapped_scores.append(score)

    sorted_scores = np.array(bootstrapped_scores)
    sorted_scores.sort()

    confidence_lower = sorted_scores[int((1.0 - ci) / 2.0 * len(sorted_scores))]
    confidence_upper = sorted_scores[int((1.0 + ci) / 2.0 * len(sorted_scores))]

    return auc, confidence_lower, confidence_upper


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
val_dataset = MedicalImageDatasetBalancedIntensity3D(csv_path=config['data']['val_csv'], root_dir= config["data"]["root_dir"])
val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, collate_fn=custom_collate, num_workers = 1)


# load the model 
backbonenet = dummy_BackboneNetV2()
mergingnetwork = InverseMergingNetworkWithClassification()
classifier = Classifier(d_model=2048)
model = MedicalTransformerLSTM(backbonenet, mergingnetwork, classifier) 
print("model loaded!!")

# load checkpoints 
pretrained_dict = torch.load(config["infer"]["checkpoints"])
model.load_state_dict(pretrained_dict["model_state_dict"])
print("checkpoint loaded")
model = model.to(device)
model.eval()


# Initialize a DataFrame to store results
results_df = pd.DataFrame(columns=['ModelOutput', 'GT', 'Predictions'])

all_labels = []
all_predictions = []
all_probabilities = []

# Inference
with torch.no_grad():
    for sample in tqdm(val_loader, desc="Validation", unit="batch"):
        
        inputs = sample['image'].to(device)
        labels = sample['label'].float().to(device)
        outputs = model(inputs)
        probabilities = torch.sigmoid(outputs).cpu().numpy()
        predictions = probabilities.round()

        all_labels.extend(labels.cpu().numpy().flatten())
        all_predictions.extend(predictions.flatten())
        all_probabilities.extend(probabilities.flatten())
       
        # Append results to the DataFrame using pd.concat
        result = pd.DataFrame({
            'ModelOutput': [probabilities[0][0]],
            'GT': [int(labels.cpu().numpy()[0])],
            'Predictions': [predictions[0][0]]
        })
        results_df = pd.concat([results_df, result], ignore_index=True)

all_labels = np.array(all_labels)
all_probabilities = np.array(all_probabilities)

# Evaluate the model with the default threshold (0.5)
default_threshold = 0.5
all_predictions_default = (all_probabilities >= default_threshold).astype(int)
auc_roc, ci_lower, ci_upper = compute_auc_with_ci(all_labels, all_probabilities)
class_report_default = classification_report(all_labels, all_predictions_default, target_names=['Class 0', 'Class 1'])
cm_default = confusion_matrix(all_labels, all_predictions_default)
sensitivity_default = cm_default[1, 1] / (cm_default[1, 1] + cm_default[1, 0])
specificity_default = cm_default[0, 0] / (cm_default[0, 0] + cm_default[0, 1])

print("Metrics with default threshold (0.5):")
print(f"AUC ROC: {auc_roc} (95% CI: {ci_lower} - {ci_upper})")
print("Classification Report:\n", class_report_default)
print(f"Sensitivity: {sensitivity_default}, Specificity: {specificity_default}")

# Compute calibration curve
prob_true, prob_pred = calibration_curve(all_labels, all_probabilities, n_bins=5)

# Plot calibration curve
plt.figure()
plt.plot(prob_pred, prob_true, marker='o', label='Model Calibration')
plt.plot([0, 1], [0, 1], linestyle='--', label='Perfect Calibration')
plt.xlabel('Predicted Probability')
plt.ylabel('True Probability')
plt.title('Calibration Plot')
plt.legend()
plt.grid(True)

# Save plot and output csv
plt.savefig('/analysis_csvs/calibration_plot.png')
plt.show()
results_df.to_csv('/analysis_csvs/infer_results.png', index=False)













