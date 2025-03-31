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
    parser.add_argument("--bert_model", type=str, default="bert-base-uncased", help="BERT model name or path used for distillation (as we'll use its tokenizer)")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the trained model")
    parser.add_argument("--max_seq_length", type=int, default=512, help="Maximum sequence length for LSTM")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training and evaluation")
    parser.add_argument("--num_classes", type=int, required=True, help="Number of classes for classification")
    parser.add_argument("--text_column", type=str, default="text", help="Column name for text data")
    parser.add_argument("--label_column", type=str, default="label", help="Column name for labels")
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

    model_state = torch.load(args.model_path)

    # Load data first
    train_data, val_data, test_data = load_data(
        args.data_path,
        text_col=args.text_column,
        label_col=args.label_column,
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
        tokenizer_name=args.bert_model,
        max_length=args.max_seq_length,
        batch_size=args.batch_size,
        num_classes=args.num_classes,
        return_datasets=True
    )

    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    # Load model
    model = DocumentBiLSTM(vocab_size=test_dataset.tokenizer.vocab_size,
                           embedding_dim=args.embedding_dim,
                           hidden_dim=args.hidden_dim,
                           n_layers=args.num_layers,
                           output_dim=args.num_classes)

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
            probs = F.softmax(outputs, dim=1)
            predictions = torch.argmax(probs, dim=1)

            all_predictions = np.append(all_predictions, predictions.cpu().numpy())

            if args.print_predictions:
                for i in range(len(predictions)):
                    print(f"Text: {test_dataset.get_text_(batch_count * args.batch_size + i)}, Prediction: {class_names[predictions[i]]}, True Label: {class_names[labels[i]]}")
                
            if args.inference_batch_limit > 0 and batch_count >= args.inference_batch_limit:
                break

            batch_count += 1

    print("Label distribution in test set:", np.bincount(all_labels))
    print("Prediction distribution:", np.bincount(all_predictions))
    print("Example predictions (first 10):", all_predictions[:10])
    print("Example true labels (first 10):", all_labels[:10])

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
            f.write(f"Predicted Class: {class_names[all_predictions[idx]] if len(class_names) > all_predictions[idx] else 'Unknown'}, True Class: {class_names[all_labels[idx]] if len(class_names) > all_labels[idx] else 'Unknown'}\n")
            f.write("\n")

    with open("metrics_lstm.txt", "w") as f:
        f.write(f"Accuracy: {accuracy}\n")
        f.write(f"F1 Score: {f1}\n")
        f.write(f"Precision: {precision}\n")
        f.write(f"Recall: {recall}\n")
            
