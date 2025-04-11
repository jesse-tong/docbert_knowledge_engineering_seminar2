import os
import pandas as pd

def merge_csv_files(directories, output_file, target_name):
    dfs = []
    for directory in directories:
        csv_path = os.path.join(directory, target_name)
        if os.path.isfile(csv_path):
            dfs.append(pd.read_csv(csv_path))
    if not dfs:
        print(f"No {target_name} found.")
        return
    common_columns = set(dfs[0].columns)
    for df in dfs[1:]:
        common_columns &= set(df.columns)
    merged = pd.concat([df[list(common_columns)] for df in dfs], ignore_index=True)
    # Shuffle the merged dataframe
    merged = merged.sample(frac=1).reset_index(drop=True)
    merged.to_csv(output_file, index=False)

if __name__ == "__main__":
    directories = ["../datasets_vithsd", "../datasets_vihsd_gemini" , "../datasets_voz_gemini"]
    merge_csv_files(directories, "../datasets/train.csv", "train.csv")
    merge_csv_files(directories, "../datasets/dev.csv", "dev.csv")
    merge_csv_files(directories, "../datasets/test.csv", "test.csv")