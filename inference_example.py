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
    parser.add_argument("--max_seq_length", type=int, default=250, help="Maximum sequence length for BERT (e.g., 250 for PhoBERT as PhoBERT allows max_position_embeddings=258)")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training and evaluation")
    parser.add_argument("--num_classes", type=int, required=True, help="Number of classes for classification")
    parser.add_argument("--text_column", type=str, default="text", help="Column name for text data")
    parser.add_argument("--label_column", type=str, nargs="+", help="Column name for labels")
    parser.add_argument("--class_names", type=str, nargs='+', required=False, help="List of class names for classification")
    parser.add_argument("--inference_batch_limit", type=int, default=-1, help="Limit for inference batch counts")
    parser.add_argument("--print_predictions", type=bool, default=False, help="Print predictions to console")
    args = parser.parse_args()

    class_names = args.class_names

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load data first
    label_column = args.label_column[0] if isinstance(args.label_column, list) and len(args.label_column) == 1 else args.label_column
    num_categories = len(args.label_column) if isinstance(args.label_column, list) else 1
    train_data, val_data, test_data = load_data(args.data_path, 
                                                text_col=args.text_column, 
                                                label_col=label_column,
                                                validation_split=0.0,
                                                test_split=1.0)
    train_loader, val_loader, test_loader = create_data_loaders(train_data=train_data, 
                                                                val_data=val_data, 
                                                                test_data=test_data, 
                                                                tokenizer_name=args.bert_model,
                                                                batch_size=args.batch_size, 
                                                                max_length=args.max_seq_length,
                                                                num_classes=args.num_classes)
    
    model = DocBERT(bert_model_name=args.bert_model, num_classes=args.num_classes, num_categories=num_categories)
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
            if num_categories > 1:
                batch_size, total_classes = outputs.shape
                if total_classes % num_categories != 0:
                    raise ValueError(f"Error: Number of total classes in the batch must of divisible by {num_categories}")

                classes_per_group = total_classes // num_categories
                # Group every classes_per_group values along dim=1
                reshaped = outputs.view(outputs.size(0), -1, classes_per_group)  # shape: (batch, self., classes_per_group)

                # Argmax over each group of classes_per_group
                predictions = reshaped.argmax(dim=-1)
            else:
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
            f.write("-" * 50 + "\n")

    with open("metrics.txt", "w") as f:
        f.write(f"Accuracy: {accuracy}\n")
        f.write(f"F1 Score: {f1}\n")
        f.write(f"Precision: {precision}\n")
        f.write(f"Recall: {recall}\n")
