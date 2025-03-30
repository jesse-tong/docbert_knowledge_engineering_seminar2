
## Example uses:

- Train with BERT model (train.csv is ag_news dataset with 4 classes)
```
python ./train.py --bert_model bert-base-uncased --data_path train.csv --label_column "Class Index" --text_column "Description" --epochs 4 --num_classes 4
```
- Inference with BERT model (test_data.csv is test dataset with 4 classes like ag_news)
```
python ./inference_example.py --bert_model bert-base-uncased --model_path "./bert_base_uncased/best_model.pth" --num_classes 4 --class_names "World" "Sports" "Business" "Science" --text_column "Description" --label_column "Class Index" --data_path "./test_data.csv" --inference_batch_limit 10
```

- Train LSTM model from BERT model using distillation (train dataset should be the same as distillation training dataset)
```
python ./distill_bert_to_lstm.py --bert_model bert-base-uncased --bert_model_path "./bert_base_uncased/best_model.pth" --output_dir "./docbert_lstm" --batch_size 32 --epochs 10 --data_path "./train.csv" --text_column "Description" --label_column "Class Index" --num_classes 4
```

- Inference with distilled LSTM model (test_data.csv is test dataset with 4 classes like ag_news)
```
python ./inference_lstm.py --model_path "./docbert_lstm/distilled_lstm_model.pth" --num_classes 4 --class_names "World" "Sports" "Business" "Science" --text_column "Description" --label_column "Class Index" --data_path "./test_data.csv" --inference_batch_limit 10 --tokenizer_path "./docbert_lstm/tokenizer.json"
```