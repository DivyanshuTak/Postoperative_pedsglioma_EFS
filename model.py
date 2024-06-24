"""
                            scripts for model blocks, and pipeline 
Author: Divyanshu Tak

Description:
    This script defines the model blocks and pipeline classes

"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
import os 
from monai.networks.nets import resnet101, resnet50, resnet18, ViT
from monai.networks.nets import UNet
from lightly.models.modules.heads import SimSiamPredictionHead, SimSiamProjectionHead
from monai.networks.nets import UNETR
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import math


"""
Fully connected Layer for classification
"""
class Classifier(nn.Module):
    def __init__(self, d_model, num_classes=2):
        super(Classifier, self).__init__()
        self.fc = nn.Linear(d_model, num_classes)
        
    def forward(self, x):
        return self.fc(x) #, x


"""
Resnet18 Backbone, cutoff before the FC layer 
"""
class dummy_BackboneNetV2(nn.Module):
    def __init__(self):
        super(dummy_BackboneNetV2, self).__init__()

        resnet = resnet18(pretrained=False)  # assuming you're not using a pretrained model
        resnet.conv1 = nn.Conv3d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        hidden_dim = resnet.fc.in_features
        self.backbone = resnet
        self.backbone.fc = nn.Identity()

    def forward(self, x):
        x = self.backbone(x)
        return x
    

"""
merging networks block : MHSA module appended by a LSTM module 
"""
class InverseMergingNetworkWithClassification(nn.Module):
    def __init__(self, feature_dim=2048, hidden_dim=512, use_attention=True):
        super(InverseMergingNetworkWithClassification, self).__init__()
        
        if use_attention:
            self.attention = nn.MultiheadAttention(feature_dim, num_heads=8)
        else:
            self.attention = False
        self.lstm = nn.LSTM(feature_dim, hidden_dim, batch_first=True, num_layers=1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        if self.attention:
            x = x.permute(1, 0, 2) 
            attn_output, _ = self.attention(x, x, x)
            x = attn_output.permute(1, 0, 2)  
 
        lstm_output, (hidden_state, _) = self.lstm(x)
        dropped = self.dropout(lstm_output)
        output_after_last_step = dropped[:, -1, :]
        relu_output = self.relu(output_after_last_step)

        return relu_output, output_after_last_step
    

"""
complete pipeline class : ResNET18 + MHSA + LSTM + Classifier 
"""
class MedicalTransformerLSTM(nn.Module):
    def __init__(self, backbone, mergingnetwork, classifier):
        super(MedicalTransformerLSTM, self).__init__()
        self.backbone = backbone
        self.classifier = classifier
        self.merging_network = mergingnetwork

        
    def forward(self, x):
        batch_size, num_scans, _, _, _ = x.shape
        x = [self.backbone(scan) for scan in x.split(1, dim=1)]
        x = torch.stack(x, dim=1).squeeze(2)
        x_preattn = x
        x,x_all = self.merging_network(x)
        x_lstm = x_all*1
        x = self.classifier(x)
        return x
    


if __name__ == "__main__":
    pass