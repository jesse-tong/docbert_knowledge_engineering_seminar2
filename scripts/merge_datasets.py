import os
import glob
import pandas as pd
import argparse
from tqdm import tqdm
import shutil

def merge_datasets(input_dirs, output_dir, preserve_splits=False):
    """
    Merge CSV datasets from multiple directories into one directory
    
    Args:
        input_dirs (list): List of input directory paths
        output_dir (str): Output directory path
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Define the expected columns for the format in test.csv
    expected_columns = ['content', 'individual', 'groups', 'religion/creed', 'race/ethnicity', 'politics']
    
    # Dictionary to hold dataframes for each split if preserving splits
    combined_data = {}
    if preserve_splits:
        combined_data = {'train': [], 'dev': [], 'test': []}
    else:
        combined_data['all'] = []
    
    # Process each input directory
    for input_dir in input_dirs:
        print(f"Processing directory: {input_dir}")
        
        # Find all CSV files in the directory
        csv_files = glob.glob(os.path.join(input_dir, "*.csv"))
        
        for file_path in tqdm(csv_files, desc=f"Processing files in {os.path.basename(input_dir)}"):
            file_name = os.path.basename(file_path)
            
            # Read the CSV file
            try:
                df = pd.read_csv(file_path)
                print(f"  Reading {file_name}: {len(df)} rows")
            except Exception as e:
                print(f"  Error reading {file_name}: {e}")
                continue
            
            # Rename 'free_text' column to 'content' if it exists
            if 'free_text' in df.columns:
                df.rename(columns={'free_text': 'content'}, inplace=True)
            
            # Check if 'content' column exists
            if 'content' not in df.columns:
                print(f"  Warning: 'content' column not found in {file_name}. Skipping.")
                continue
            
            # Ensure all required columns exist
            for col in expected_columns:
                if col != 'content' and col not in df.columns:
                    df[col] = 0  # Set default value for missing columns
            
            # Convert category columns to integer type
            for col in expected_columns:
                if col != 'content' and col in df.columns:
                    df[col] = df[col].fillna(0).astype(int)
            
            # Drop unnecessary columns
            df = df[expected_columns]
            
            # Determine which split this file belongs to
            if preserve_splits:
                if 'train' in file_name.lower():
                    combined_data['train'].append(df)
                elif 'dev' in file_name.lower():
                    combined_data['dev'].append(df)
                elif 'test' in file_name.lower():
                    combined_data['test'].append(df)
                else:
                    # If not explicitly marked, add to all splits
                    for split in ['train', 'dev', 'test']:
                        combined_data[split].append(df)
            else:
                combined_data['all'].append(df)
    
    # Combine and save the data
    for split, dfs in combined_data.items():
        if not dfs:
            print(f"No data for {split} split")
            continue
            
        combined_df = pd.concat(dfs, ignore_index=True)
        
        # Remove duplicates
        combined_df = combined_df.drop_duplicates(subset=['content'])
        
        # Save to output directory
        output_file = os.path.join(output_dir, f"{split}.csv" if preserve_splits else "combined.csv")
        combined_df.to_csv(output_file, index=False)
        print(f"Saved {len(combined_df)} rows to {output_file}")

def main():
    parser = argparse.ArgumentParser(description="Merge CSV datasets from multiple directories")
    parser.add_argument("--input_dirs", required=True, nargs='+', 
                        help="List of input directory paths containing CSV files")
    parser.add_argument("--output_dir", required=True, 
                        help="Output directory path for merged datasets")
    
    args = parser.parse_args()
    
    merge_datasets(
        args.input_dirs, 
        args.output_dir, 
        preserve_splits=True
    )

if __name__ == "__main__":
    main()