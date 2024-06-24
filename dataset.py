"""
                            scripts for dataset, and input augmentations
Author: Divyanshu Tak

Description:
    This script defines dataset classes, and augmentation funtions for longitudinal imaging trainign

"""

import os
import torch
import monai
import numpy as np
import pandas as pd 
from monai.transforms import (
    Compose, LoadImaged, ScaleIntensityd, ToTensord,
    RandFlipd, RandRotated
)
from torch.utils.data import Dataset
from monai.utils import set_determinism
import nibabel as nib 
from monai.transforms import Compose, Affined, RandGaussianNoised, Rand3DElasticd, AdjustContrastd, ScaleIntensityd, ToTensord, Resized, RandRotate90d, Resize, RandGaussianSmoothd, GaussianSmoothd, Rotate90d
import random 
from torch.nn.functional import one_hot
from sklearn.preprocessing import LabelEncoder
import numpy as np
from monai.transforms import (
    Compose, LoadImaged, Resized, RandRotate90d, RandAffined, ScaleIntensityd, 
    Rand3DElasticd, RandGaussianNoised, NormalizeIntensityd, ToTensord
)
from scipy.ndimage import histogram

    
###################################################
##      3D TRANSFORM FOR VALIDATION/TEST LOADERS
###################################################

class NormalSynchronizedTransform3D:
    def __init__(self, image_size=(96,128,96), max_rotation=40, translate_range=0.2, scale_range=(0.9, 1.3), apply_prob=0.5):
        self.image_size = image_size
        self.max_rotation = max_rotation
        self.translate_range = translate_range
        self.scale_range = scale_range
        self.apply_prob = apply_prob

    def __call__(self, scan_list):
        transformed_scans = []
        resize_transform = Resized(spatial_size=(96,128,96), keys=["image"])
        scale_transform = ScaleIntensityd(keys=["image"], minv=0.0, maxv=1.0)  # Intensity scaling
        tensor_transform = ToTensord(keys=["image"])  # Convert to tensor

        for scan in scan_list:
            sample = {"image": scan}
            sample = resize_transform(sample)
            sample = scale_transform(sample)
            sample = tensor_transform(sample)
            transformed_scans.append(sample["image"].squeeze())
            
        return torch.stack(transformed_scans)

# VALIDATION CLASS 
class MedicalImageDatasetBalancedIntensity3D(Dataset):
    def __init__(self, csv_path, root_dir, transform=None):
        self.dataframe = pd.read_csv(csv_path)
        self.root_dir = root_dir
        self.transform = NormalSynchronizedTransform3D()

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        pat_id = self.dataframe.loc[idx, 'pat_id']
        scan_dates = str(self.dataframe.loc[idx, 'scandate'])
        label = self.dataframe.loc[idx, 'label']
        scandates = scan_dates.split('-')
        scan_list = []

        for scandate in scandates:
            img_name = os.path.join(self.root_dir, f"{pat_id}_{scandate}.nii.gz") 
            scan = nib.load(img_name).get_fdata()
            scan_list.append(torch.tensor(scan, dtype=torch.float32).unsqueeze(0))

        transformed_scans = self.transform(scan_list)
        sample = {"image": transformed_scans, "label": torch.tensor(label, dtype=torch.float32)}
        return sample

###################################################
##      3D TRANSFORM FOR TRAIN LOADERS
###################################################
"""
the transform parameters must remain same for all the images in a single input list 
"""
class SynchronizedTransform3D:
    def __init__(self, image_size=(96,128,96), max_rotation=0.34, translate_range=15, scale_range=(0.9, 1.3), apply_prob=0.5, gaussian_sigma_range=(0.25, 1.5), gaussian_noise_std_range=(0.05, 0.09)):
        self.image_size = image_size
        self.max_rotation = max_rotation
        self.translate_range = translate_range
        self.scale_range = scale_range
        self.apply_prob = apply_prob
        self.gaussian_sigma_range = gaussian_sigma_range
        self.gaussian_noise_std_range = gaussian_noise_std_range

    def __call__(self, scan_list):
        transformed_scans = []
        rotate_params = (random.uniform(-self.max_rotation, self.max_rotation),) * 3 if random.random() < self.apply_prob else (0, 0, 0)
        translate_params = tuple([random.uniform(-self.translate_range, self.translate_range) for _ in range(3)]) if random.random() < self.apply_prob else (0, 0, 0)
        scale_params = tuple([random.uniform(self.scale_range[0], self.scale_range[1]) for _ in range(3)]) if random.random() < self.apply_prob else (1, 1, 1)
        gaussian_sigma = tuple([random.uniform(self.gaussian_sigma_range[0], self.gaussian_sigma_range[1]) for _ in range(3)]) if random.random() < self.apply_prob else None
        gaussian_noise_std = random.uniform(self.gaussian_noise_std_range[0], self.gaussian_noise_std_range[1]) if random.random() < self.apply_prob else None
        flip_axes = None#(0,1) if random.random() < self.apply_prob else None  # Determine if and along which axes to flip

        affine_transform = Affined(keys=["image"], rotate_params=rotate_params, translate_params=translate_params, scale_params=scale_params, padding_mode='zeros')
        gaussian_blur_transform = GaussianSmoothd(keys=["image"], sigma=gaussian_sigma) if gaussian_sigma else None
        gaussian_noise_transform = RandGaussianNoised(keys=["image"], std=gaussian_noise_std, prob=1.0, mean=0.0, sample_std=False) if gaussian_noise_std else None
        flip_transform = Rotate90d(keys=["image"], k=1, spatial_axes=flip_axes) if flip_axes else None
        resize_transform = Resized(spatial_size=(96,128,96), keys=["image"])
        scale_transform = ScaleIntensityd(keys=["image"], minv=0.0, maxv=1.0)  
        tensor_transform = ToTensord(keys=["image"]) 

        for scan in scan_list:
            sample = {"image": scan}
            sample = resize_transform(sample)
            sample = affine_transform(sample)
            if flip_transform:
                sample = flip_transform(sample)
            if gaussian_blur_transform:
                sample = gaussian_blur_transform(sample)
            sample = scale_transform(sample)
            if gaussian_noise_transform:
                sample = gaussian_noise_transform(sample)
            sample = tensor_transform(sample)
            transformed_scans.append(sample["image"].squeeze())
            
        return torch.stack(transformed_scans)

# TRAIN CLASS 
class TransformationMedicalImageDatasetBalancedIntensity3D(Dataset):
    def __init__(self, csv_path, root_dir, transform=None):
        self.dataframe = pd.read_csv(csv_path)
        self.root_dir = root_dir
        self.transform = SynchronizedTransform3D()

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        pat_id = self.dataframe.loc[idx, 'pat_id']
        scan_dates = str(self.dataframe.loc[idx, 'scandate'])
        label = self.dataframe.loc[idx, 'label']
        scandates = scan_dates.split('-')
        scan_list = []

        for scandate in scandates:
            img_name = os.path.join(self.root_dir, f"{pat_id}_{scandate}.nii.gz")
            scan = nib.load(img_name).get_fdata()
            scan_list.append(torch.tensor(scan, dtype=torch.float32).unsqueeze(0))

        transformed_scans = self.transform(scan_list)
        sample = {"image": transformed_scans, "label": torch.tensor(label, dtype=torch.float32)}
        return sample


if __name__ == "__main__":

    pass