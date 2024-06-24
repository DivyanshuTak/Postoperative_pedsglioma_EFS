"""
                             oversampling csv for temporal learning 
Author: Divyanshu Tak

Description:
    This script reads the longitduinal csv, and generates the oversampled csv 
    for temoral learning trainign and validation 

Usage:
    Run the script with the path to the input CSV file as an argument.
    Example: python script_name.py input.csv
"""

import pandas as pd
import argparse
from itertools import permutations
def parse_scandates(scandates_str):
    """Parse the scandates from the string as strings."""
    return scandates_str.split('-')

def is_chronological(dates):
    """Check if dates (in string format assumed to be YYYYMMDD or similar) are in chronological order."""
    return all(dates[i] <= dates[i + 1] for i in range(len(dates) - 1))

def generate_oversampled_data2(csv_path):
    import pandas as pd
    from itertools import permutations

    df = pd.read_csv(csv_path, dtype={"pat_id": str, "scandate": str})
    df.columns = ['pat_id', 'scandate', 'label'] 
    
    oversampled_data = []
    
    for index, row in df.iterrows():
        scandates = parse_scandates(row['scandate'])
        max_length = min(5, len(scandates)) 
        patient_samples = 0
        
        for r in range(2, max_length + 1):
            for perm in permutations(scandates, r):
                if patient_samples >= 700:
                    break
                label = 1 if is_chronological(perm) else 0
                oversampled_data.append([row['pat_id'], '-'.join(perm), label])
                patient_samples += 1
            if patient_samples >= 700:
                break

    oversampled_df = pd.DataFrame(oversampled_data, columns=['pat_id', 'scandate', 'label'])
    
    # Balancing the labels to 50:50
    min_label_count = min(oversampled_df[oversampled_df['label'] == 1].shape[0], 
                          oversampled_df[oversampled_df['label'] == 0].shape[0])
    
    label1_samples = oversampled_df[oversampled_df['label'] == 1]#.sample(n=min_label_count, random_state=1)
    label0_samples = oversampled_df[oversampled_df['label'] == 0]#.sample(n=min_label_count, random_state=1)
    
    final_df = pd.concat([label1_samples, label0_samples])
    final_df = final_df.sample(frac=1).reset_index(drop=True) 
    
    oversampled_csv_path = args.output_path
    final_df.to_csv(oversampled_csv_path, index=False)
    
    # Print label percentages
    final_label1_percentage = (final_df['label'] == 1).mean() * 100
    final_label0_percentage = (final_df['label'] == 0).mean() * 100
    
    print(f"Percentage of Label 1: {final_label1_percentage:.2f}%")
    print(f"Percentage of Label 0: {final_label0_percentage:.2f}%")
    
    return oversampled_csv_path


# function call 
parser = argparse.ArgumentParser(description='MRI registration and processing') 
parser.add_argument('input_path', type=str, help='Path to the input csv file')
parser.add_argument('output_path', type=str, help='Path to the output csv file')
args = parser.parse_args()

oversampled_csv_path = generate_oversampled_data2(args.input_path)
