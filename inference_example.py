from model import DocBERT
from dataset import load_data, create_data_loaders
from trainer import Trainer
import argparse
import os, sklearn
import numpy as np
import torch

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Document Classification with Distillation")
    parser.add_argument("--data_path", type=str, required=True, help="Path to the dataset")
    parser.add_argument("--bert_model", type=str, default="bert-base-uncased", help="Pre-trained BERT model name")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the trained model")
    parser.add_argument("--max_seq_length", type=int, default=512, help="Maximum sequence length for BERT")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training and evaluation")
    parser.add_argument("--num_classes", type=int, required=True, help="Number of classes for classification")
    parser.add_argument("--text_column", type=str, default="text", help="Column name for text data")
    parser.add_argument("--label_column", type=str, default="label", help="Column name for labels")
    parser.add_argument("--class_names", type=str, nargs='+', required=True, help="List of class names for classification")
    parser.add_argument("--inference_batch_limit", type=int, default=-1, help="Limit for inference batch counts")
    parser.add_argument("--print_predictions", type=bool, default=False, help="Print predictions to console")
    args = parser.parse_args()

    class_names = args.class_names

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_data, val_data, test_data = load_data(args.data_path, 
                                                text_col=args.text_column, 
                                                label_col=args.label_column,
                                                validation_split=0.0,
                                                test_split=1.0)
    train_loader, val_loader, test_loader = create_data_loaders(train_data=train_data, 
                                                                val_data=val_data, 
                                                                test_data=test_data, 
                                                                tokenizer_name=args.bert_model,
                                                                batch_size=args.batch_size, 
                                                                max_length=args.max_seq_length)
    
    model = DocBERT(bert_model_name=args.bert_model, num_classes=args.num_classes)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model = model.to(device)

    all_labels = np.array([], dtype=int)
    all_predictions = np.array([], dtype=int)
    batch_window_index = 0
    batch_size = args.batch_size

    # Inference
    for batch in test_loader:
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        token_type_ids = batch['token_type_ids']
        labels = batch['label']

        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        token_type_ids = token_type_ids.to(device)
        labels = labels.to(device)
        all_labels = np.append(all_labels, labels.cpu().numpy())

        with torch.no_grad():
            outputs = model(input_ids, attention_mask=attention_mask)
            logits = outputs
            predictions = torch.argmax(logits, dim=-1)
            all_predictions = np.append(all_predictions, predictions.cpu().numpy())

        if args.print_predictions:
            for i in range(len(predictions)):
                idx = int(i)
                print(f"Text: {test_data[0][batch_window_index*batch_size + idx]}")
                print(f"True Label: {labels[idx].item()}, Predicted Label: {predictions[idx].item()}")
                print(f"Predicted Class: {class_names[predictions[idx].item() if len(class_names) > predictions[idx].item() else 'Unknown']}")
                print(f"True Class: {class_names[labels[idx].item()]  if len(class_names) > predictions[idx].item() else 'Unknown'}")
                print("-" * 50)

        batch_window_index += 1
        if args.inference_batch_limit > 0 and batch_window_index >= args.inference_batch_limit:
            break

    # Calculate accuracy, F1 score, recall, and precision
    accuracy = sklearn.metrics.accuracy_score(all_labels, all_predictions)
    f1 = sklearn.metrics.f1_score(all_labels, all_predictions, average='weighted')
    precision = sklearn.metrics.precision_score(all_labels, all_predictions, average='weighted')
    recall = sklearn.metrics.recall_score(all_labels, all_predictions, average='weighted')

    print(f"Accuracy: {accuracy}")
    print(f"F1 Score: {f1}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")

    with open("predictions.txt", "w") as f:
        for i in range(len(all_labels)):
            idx = int(i)
            f.write(f"Text: {test_data[0][idx]}\n")
            f.write(f"True Label: {all_labels[idx]}, Predicted Label: {all_predictions[idx]}\n")
            f.write(f"Predicted Class: {class_names[all_predictions[idx]] if len(class_names) > all_predictions[idx] else 'Unknown'}, True Class: {class_names[all_labels[idx]] if len(class_names) > all_labels[idx] else 'Unknown'}\n")
            f.write("-" * 50 + "\n")

    with open("metrics.txt", "w") as f:
        f.write(f"Accuracy: {accuracy}\n")
        f.write(f"F1 Score: {f1}\n")
        f.write(f"Precision: {precision}\n")
        f.write(f"Recall: {recall}\n")
