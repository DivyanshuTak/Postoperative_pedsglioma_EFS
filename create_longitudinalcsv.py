"""
                             Longitudinal csv generation from image directory  
Author: Divyanshu Tak

Description:
    This script the longitudinal csv from the image directory. the images are fomatted 
    in patid_scandate.nii.gz format. the output csv has 3 columns [patid, scandate, label] 

Usage:
    Run the script with the path to the input and outout CSV file, and labels as an argument.
"""

import os
import pandas as pd
from collections import defaultdict
import argparse

def create_patient_scan_csv(directory_path, output_csv, labels):
    """
    Creates a CSV file with columns pat_id, scandate, and label from a directory of filenames.

    Args:
        directory_path (str): Path to the directory containing the .nii.gz files.
        output_csv (str): Path to the output CSV file.
        labels (list): List of labels for the output CSV.
    """
    patient_scans = defaultdict(list)
    
   
    for filename in os.listdir(directory_path):
        if filename.endswith(".nii.gz"):
            # filenames are fomatted as : patid_scandate.nii.gz
            base_name = filename[:-7]  
            pat_id, scandate = base_name.split('_')
            patient_scans[pat_id].append(scandate)
    
    
    data = []
    for i, (pat_id, scandates) in enumerate(patient_scans.items()):
        scandates_str = '-'.join(sorted(scandates))  # collate the scandates using "-"
        label = labels[i] if i < len(labels) else 'no_label'  
        data.append([pat_id, scandates_str, label])
    
    # Create a DataFrame and save to CSV
    df = pd.DataFrame(data, columns=['pat_id', 'scandate', 'label'])
    df.to_csv(output_csv, index=False)

def main():
    parser = argparse.ArgumentParser(description='Create longitudinal csv from image directory')
    parser.add_argument('directory_path', type=str, help='Path to the directory containing the nifti files.')
    parser.add_argument('output_csv', type=str, help='Path to the output CSV file.')
    parser.add_argument('labels', type=str, help=' list of labels')

    args = parser.parse_args()
    
    # Convert the labels argument to a list
    labels = args.labels.split(',')

    create_patient_scan_csv(args.directory_path, args.output_csv, labels)

if __name__ == "__main__":
    main()
