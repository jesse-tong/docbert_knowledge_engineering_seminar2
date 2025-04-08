
## Example uses:

- Train with BERT model (train.csv is ViTHSD dataset with 4 classes each for 5 categories)
```
python ./train.py --bert_model "vinai/phobert-base-v2" --train_data_path "./datasets/train.csv" --val_data_path "./datasets/dev.csv" --test_data_path "./datasets/test.csv" --label_column "individual" "groups" "religion/creed" "race/ethnicity" "politics" --text_column "content" --epochs 7 --num_classes 4 --output "./vietnamese_hate_speech_detection_phobert"
```
- Inference with BERT model (test_data.csv is test dataset with 4 classes each for 5 categories like ViTHSD)
```
python ./inference_example.py --bert_model "vinai/phobert-base-v2" --model_path "./vietnamese_hate_speech_detection_phobert/vinai_phobert-base-v2_finetuned.pth" --num_classes 4  --label_column "individual" "groups" "religion/creed" "race/ethnicity" "politics" --text_column "content" --data_path "./datasets/test.csv" --inference_batch_limit 10
```

- Train LSTM model from BERT model using distillation (train dataset should be the same as distillation training dataset)
```
python ./distill_bert_to_lstm.py --bert_model "vinai/phobert-base-v2" --bert_model_path "./vietnamese_hate_speech_detection_phobert/vinai_phobert-base-v2_finetuned.pth" --output_dir "./vietnamese_hate_speech_detection_phobert" --batch_size 32 --epochs 10 --train_data_path "./datasets/train.csv" --val_data_path "./datasets/dev.csv" --test_data_path "./datasets/test.csv" --label_column "individual" "groups" "religion/creed" "race/ethnicity" "politics" --text_column "content" --num_classes 4
```

- Inference with distilled LSTM model (test_data.csv is test dataset with 4 classes like ag_news)
```
python ./inference_lstm.py --model_path "./vietnamese_hate_speech_detection_phobert/distilled_lstm_model.pth" --bert_tokenizer "vinai/phobert-base-v2" --num_classes 4  --label_column "individual" "groups" "religion/creed" "race/ethnicity" "politics" --text_column "content" --data_path "./datasets/test.csv" --inference_batch_limit 10
```

## How to run:

- Install the dependencies in requirements.txt: pip install -r requirements.txt

- Either follow the "Train with BERT model" or "Train LSTM model from BERT model using distillation" in Example uses section above, or git clone the model from: "https://huggingface.co/jesse-tong/vietnamese_hate_speech_detection_phobert"

- Run the Streamlit app: streamlit run app.py, then either go to http://localhost:8501 or waiting for the browser tab to open.