from dataset import load_data, create_data_loaders
from models.lstm_model import DocumentBiLSTM
from sklearn import metrics
import torch, random
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
import argparse

# Add these imports for mapping optimization
from itertools import permutations
import copy

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Document Classification with LSTM")
    parser.add_argument("--data_path", type=str, required=True, help="Path to the dataset")
    parser.add_argument("--bert_tokenizer", type=str, default="bert-base-uncased", help="BERT model name or path used for distillation (as we'll use its tokenizer)")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the trained model")
    parser.add_argument("--max_seq_length", type=int, default=250, help="Maximum sequence length for LSTM")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training and evaluation")
    parser.add_argument("--num_classes", type=int, required=True, help="Number of classes for classification")
    parser.add_argument("--text_column", type=str, default="text", help="Column name for text data")
    parser.add_argument("--label_column", type=str, nargs='+', help="Column name for labels")
    parser.add_argument("--class_names", type=str, nargs='+', required=True, help="List of class names for classification")
    parser.add_argument("--inference_batch_limit", type=int, default=-1, help="Limit for inference batch counts")
    parser.add_argument("--print_predictions", type=bool, default=False, help="Print predictions to console")

    # LSTM model arguments
    parser.add_argument("--embedding_dim", type=int, default=300, help="Dimension of word embeddings in LSTM")
    parser.add_argument("--hidden_dim", type=int, default=256, help="Hidden dimension of LSTM")
    parser.add_argument("--num_layers", type=int, default=2, help="Number of LSTM layers")
    parser.add_argument("--dropout", type=float, default=0.5, help="Dropout probability")
    args = parser.parse_args()

    class_names = args.class_names

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_state = torch.load(args.model_path, map_location=device)

    # Load data first
    label_column = args.label_column[0] if isinstance(args.label_column, list) and len(args.label_column) == 1 else args.label_column
    num_categories = len(args.label_column) if isinstance(args.label_column, list) else 1
    train_data, val_data, test_data = load_data(
        args.data_path,
        text_col=args.text_column,
        label_col=label_column,
        validation_split=0.0,
        test_split=1.0,
        seed=42
    )
    
    # Create BERT data loaders
    print("Creating data loaders (note the datasets and dataloaders use BERT's tokenizer)...")
    train_dataset, val_dataset, test_dataset = create_data_loaders(
        train_data, 
        val_data, 
        test_data,
        tokenizer_name=args.bert_tokenizer,
        max_length=args.max_seq_length,
        batch_size=args.batch_size,
        num_classes=args.num_classes,
        return_datasets=True
    )

    bert_vocab_size = train_dataset.tokenizer.vocab_size
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    # Load model
    model = DocumentBiLSTM(vocab_size=bert_vocab_size,
                           embedding_dim=args.embedding_dim,
                           hidden_dim=args.hidden_dim,
                           n_layers=args.num_layers,
                           output_dim=args.num_classes * num_categories)
    
    if 'model_state_dict' in model_state:
        model.load_state_dict(model_state['model_state_dict'], strict=False)
    else:
        model.load_state_dict(model_state, strict=False)

    model = model.to(device)

    all_labels = np.array([], dtype=int)
    all_predictions = np.array([], dtype=int)

    # Inference
    batch_count = 0
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch['input_ids'].to(device)
            labels = batch['label'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            all_labels = np.append(all_labels, labels.cpu().numpy())

            outputs = model(input_ids, attention_mask=attention_mask)
            
            if num_categories > 1:
                batch_size, total_classes = outputs.shape
                if total_classes % num_categories != 0:
                    raise ValueError(f"Error: Number of total classes in the batch must of divisible by {num_categories}")

                classes_per_group = total_classes // num_categories
                # Group every classes_per_group values along dim=1
                reshaped = outputs.view(outputs.size(0), -1, classes_per_group)  # shape: (batch, num_categories, classes_per_group)
                probs = F.softmax(reshaped, dim=1)
                # Argmax over each group of classes_per_group
                print("DEBUG: Reshaped shape: ", reshaped.shape)

                predictions = torch.argmax(probs, dim=1)
            else:
                probs = F.softmax(outputs, dim=1)
                predictions = torch.argmax(probs, dim=1)

            print("DEBUG: Prediction shape: ", predictions.shape)

            all_predictions = np.append(all_predictions, predictions.cpu().numpy())

            if args.print_predictions:
                for i in range(len(predictions)):
                    print(f"Text: {test_dataset.get_text_(batch_count * args.batch_size + i)}, Prediction: {predictions[i]}, True Label: {labels[i]}")
                
            if args.inference_batch_limit > 0 and batch_count >= args.inference_batch_limit:
                break

            batch_count += 1

    # Turn predictions and labels to 1D arrays
    all_labels = all_labels.reshape(-1, 1)
    all_labels = np.array([int(label) for label in all_labels])
    all_predictions = all_predictions.reshape(-1, 1)
    print("DEBUG: all_labels shape: ", all_labels.shape)
    print("DEBUG: all_predictions shape: ", all_predictions.shape)
    # Print classification report
    # Calculate accuracy, F1 score, recall, and precision
    accuracy = metrics.accuracy_score(all_labels, all_predictions)
    f1 = metrics.f1_score(all_labels, all_predictions, average='weighted')
    precision = metrics.precision_score(all_labels, all_predictions, average='weighted')
    recall = metrics.recall_score(all_labels, all_predictions, average='weighted')

    print(f"Accuracy: {accuracy}")
    print(f"F1 Score: {f1}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")

    with open("predictions_lstm.txt", "w") as f:
        for i in range(len(all_labels)):
            idx = int(i)
            f.write(f"Text: {test_dataset.get_text_(idx)}\n")
            f.write(f"True Label: {all_labels[idx]}, Predicted Label: {all_predictions[idx]}\n")
            f.write("\n")

    with open("metrics_lstm.txt", "w") as f:
        f.write(f"Accuracy: {accuracy}\n")
        f.write(f"F1 Score: {f1}\n")
        f.write(f"Precision: {precision}\n")
        f.write(f"Recall: {recall}\n")
            
