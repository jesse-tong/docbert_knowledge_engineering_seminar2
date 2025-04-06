
## Example uses:

- Train with BERT model (train.csv is ViTHSD dataset with 4 classes each for 5 categories)
```
python ./train.py --bert_model "vinai/phobert-base-v2" --train_data_path "./datasets/train.csv" --val_data_path "./datasets/dev.csv" --test_data_path "./datasets/test.csv" --label_column "individual" "groups" "religion/creed" "race/ethnicity" "politics" --text_column "content" --epochs 7 --num_classes 4
```
- Inference with BERT model (test_data.csv is test dataset with 4 classes each for 5 categories like ViTHSD)
```
python ./inference_example.py --bert_model "vinai/phobert-base-v2" --model_path "./vinai_phobert-base-v2_finetuned/best_model.pth" --num_classes 4  --label_column "individual" "groups" "religion/creed" "race/ethnicity" "politics" --text_column "content" --data_path "./datasets/test.csv" --inference_batch_limit 10
```

- Train LSTM model from BERT model using distillation (train dataset should be the same as distillation training dataset)
```
python ./distill_bert_to_lstm.py --bert_model "vinai/phobert-base-v2" --bert_model_path "./vinai_phobert-base-v2_finetuned/best_model.pth" --output_dir "./docbert_lstm" --batch_size 32 --epochs 10 --train_data_path "./datasets/train.csv" --val_data_path "./datasets/dev.csv" --test_data_path "./datasets/test.csv" --label_column "individual" "groups" "religion/creed" "race/ethnicity" "politics" --text_column "content" --num_classes 4
```

- Inference with distilled LSTM model (test_data.csv is test dataset with 4 classes like ag_news)
```
python ./inference_lstm.py --model_path "./docbert_lstm/distilled_lstm_model.pth" --num_classes 4  --label_column "individual" "groups" "religion/creed" "race/ethnicity" "politics" --text_column "content" --data_path "./dataset/test.csv" --inference_batch_limit 10
```