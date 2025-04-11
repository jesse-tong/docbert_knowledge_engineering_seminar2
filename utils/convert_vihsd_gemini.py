import os
import glob
import pandas as pd
import argparse
from google import genai
from tqdm import tqdm
import time
import re
from word_segmentation_vi import word_segmentation_vi

def setup_genai(api_key):
    """Configure the Google Generative AI client with your API key"""
    return genai.Client(api_key=api_key)

def classify_text(model, text, suggest_label=False):
    """Classify Vietnamese text into hate speech categories using Google's Generative AI"""
    prompt = f"""
    Analyze the following Vietnamese text for hate speech (each sentence is separated by a newline):
    "{text}"
    
    Rate it on these categories (0=NORMAL, 1=CLEAN, 2=OFFENSIVE, 3=HATE):
    - individual (targeting specific individuals)
    - groups (targeting groups or organizations)
    - religion/creed (targeting religious groups or beliefs)
    - race/ethnicity (racial/ethnic hate speech)
    - politics (political hate speech)
    If the text doesn't specify a person or group in a category, return 0 for that category.
    Else, return 1 for CLEAN, 2 for OFFENSIVE, or 3 for HATE.

    {'The number at the end of the sentence (between <SuggestLabel> and </SuggestLabel> tags is the suggestion label for the sentence. (0 is normal/clean, 1 is offensive/hate in at least one category)' if suggest_label else ''}
    
    For each sentence in the text, return only 5 numbers separated by commas (corresponding to the label of individual, groups, religion/creed, race/ethnicity, politics) and numbers for each sentence seperated by newlines, like (with no other text): 
    0,1,0,0,0
    1,0,0,0,2
    """
    
    try:
        response = model.models.generate_content(model="gemini-2.0-flash", contents=prompt)
        values = response.text.strip().split('\n')
        values = [line.split(',') for line in values]
        return values
        
    except Exception as e:
        print(f"Error classifying text: {e}")
        return None

def process_file(input_file, output_file, model, rate_limit_pause=4, text_col="free_text", suggest_column="labels"):
    """Process a single CSV file to match the test.csv format"""
    print(f"Processing {input_file}...")
    
    # Read the input file
    try:
        df = pd.read_csv(input_file)
    except Exception as e:
        print(f"Error reading {input_file}: {e}")
        return
    
    # Rename column text_col to content
    if text_col in df.columns:
        df.rename(columns={text_col: 'content'}, inplace=True)
    elif 'content' not in df.columns:
        print(f"Error: 'content' column not found in {input_file}")
        return
    
    # Ensure all required columns exist
    category_columns = ['individual', 'groups', 'religion/creed', 'race/ethnicity', 'politics']
    for col in category_columns:
        if col not in df.columns:
            # Change column type to int if it doesn't exist
            df[col] = 0

    print("Suggesting labels: ", 'True' if suggest_column in df.columns else 'False')
    
    # Process each batch (100 rows at a time)
    batch_size = 100
    for start in tqdm(range(0, len(df), batch_size), desc="Processing batches"):
        end = min(start + batch_size, len(df))
        batch_df = df.iloc[start:end]
        
        # Skip if all categories already have values
        if all(batch_df[cat].all() != 0 for cat in category_columns):
            continue
        
        # Join 50 rows by newlines, and classify all at once
        batch_strings = [str(sentence) for sentence in batch_df['content'].tolist()]
        suggest_label = False
        if suggest_column in df.columns:
            batch_strings = [str(sentence) + " " + f"<SuggestLabel>{str(label)}</SuggestLabel>" for sentence, label in zip(batch_strings, batch_df[suggest_column].tolist())]
            suggest_label = True


        text_to_classify = "\n".join(batch_strings)
        classifications = classify_text(model, text_to_classify, suggest_label=suggest_label)


        # Try 2 more times, else skip
        if classifications is None:
            for _ in range(2):
                classifications = classify_text(model, text_to_classify)
                if classifications is not None:
                    break
                time.sleep(rate_limit_pause)
            else:
                print(f"Error classifying batch starting at index {start}. Skipping...")
                continue

        try:
            # Update the DataFrame with the classifications
            for i, row in enumerate(classifications):
                for j, col in enumerate(category_columns):
                    df.at[start + i, col] = int(row[j])
        except Exception as e:
            for _ in range(2):
                classifications = classify_text(model, text_to_classify)
                if classifications is not None:
                    break
                time.sleep(rate_limit_pause)
            else:
                print(f"Error classifying batch starting at index {start}. Skipping...")
                continue
        
        try:
            for i, row in enumerate(classifications):
                for j, col in enumerate(category_columns):
                    df.at[start + i, col] = int(row[j])
        except Exception as e:
            print(f"Error updating DataFrame: {e}")
            continue
        
        time.sleep(rate_limit_pause)
    
    # Apply word segmentation to the content column
    df['content'] = df['content'].apply(lambda x: word_segmentation_vi(str(x)))
    
    # Save processed file, export columns of category_columns is int
    for col in category_columns:
        df[col] = df[col].astype(int)
    # Drop label_id column if it exists
    if 'label_id' in df.columns:
        df.drop(columns=['label_id'], inplace=True)
    df.to_csv(output_file, index=False)
    print(f"Saved processed file to {output_file}")

def main():
    parser = argparse.ArgumentParser(description="Process ViHSD CSV files with Google Generative AI")
    parser.add_argument("--input_dir", required=True, help="Directory containing input CSV files")
    parser.add_argument("--output_dir", required=True, help="Directory to save processed files")
    parser.add_argument("--api_key", required=True, help="Google Generative AI API key")
    parser.add_argument("--pause", type=float, default=4.0, help="Pause between API calls (seconds)")
    parser.add_argument("--text_col", default="free_text", help="Column name for text content in input CSV files")
    
    args = parser.parse_args()
    
    # Ensure output directory exists
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Setup Google Generative AI
    model = setup_genai(args.api_key)
    
    # Get all CSV files in the input directory
    csv_files = glob.glob(os.path.join(args.input_dir, "*.csv"))
    if not csv_files:
        print(f"No CSV files found in {args.input_dir}")
        return
        
    print(f"Found {len(csv_files)} CSV files to process")
    
    # Process each file
    for input_file in csv_files:
        output_file = os.path.join(args.output_dir, os.path.basename(input_file))
        if os.path.exists(output_file):
            print(f"Output file {output_file} already exists. Skipping...")
            continue
        process_file(input_file, output_file, model, args.pause, text_col=args.text_col)

if __name__ == "__main__":
    # This script is used to process ViHSD CSV files with Google Generative AI
    # First, git clone from https://huggingface.co/datasets/sonlam1102/vihsd
    main()