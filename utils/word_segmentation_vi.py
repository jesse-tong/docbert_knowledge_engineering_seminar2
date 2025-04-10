from underthesea import word_tokenize
import os, pandas

def word_segmentation_vi(text):
    segmented_text = word_tokenize(text, format="text")
    return segmented_text

if __name__ == "__main__":
    # Script này để segment các file CSV và TSV trong thư mục datasets cho tiếng Việt (do PhoBERT yêu cầu đầu vào đã được segment theo từ)
    dataset_dir = "../datasets_vithsd"

    csv_files = [f for f in os.listdir(dataset_dir) if f.endswith('.csv')]
    tsv_files = [f for f in os.listdir(dataset_dir) if f.endswith('.tsv')]

    for file in csv_files:
        file_path = os.path.join(dataset_dir, file)
        df = pandas.read_csv(file_path)
        if 'content' in df.columns:
            df['content'] = df['content'].apply(lambda text: word_segmentation_vi(str(text)))

            if 'Unnamed: 0' in df.columns:
                df.drop(columns=['Unnamed: 0'], inplace=True)

            df.to_csv(file_path, index=False)
            print(f"Processed {file}")
        else:
            print(f"'content' column not found in {file}")