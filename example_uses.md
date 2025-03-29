
## Example uses:

- Train with BERT model (train.csv is ag_news dataset with 4 classes)
```
python train.py --data_path train.csv --label_column "Class Index" --text_column "Description" --epochs 4 --num_classes 4
```
- Inference with BERT model (train.csv is ag_news dataset with 4 classes)
```
python .\inference_example.py --model_path "./bert_base_uncased/best_model.pth" --num_classes 4 --class_names "World" "Sports" "Business" "Science" --text_column "Description" --label_column "Class Index" --data_path "./train.csv" --inference_batch_limit 10
```

- Train LSTM model from BERT model using distillation
```
python .\distill_bert_to_lstm.py --bert_model bert-base-uncased --bert_model_path "./bert_base_uncased/best_model.pth" --output_dir "./docbert_lstm" --batch_size 32 --epochs 10 --data_path "./train.csv" --text_column "Description" --label_column "Class Index" --num_classes 4
```