from dataset_lstm import prepare_lstm_data, LSTMTokenizer, LSTMDataset
from models.lstm_model import DocumentBiLSTM
import sklearn, torch
import numpy as np
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Document Classification with LSTM")
    parser.add_argument("--data_path", type=str, required=True, help="Path to the dataset")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the trained model")
    parser.add_argument("--max_seq_length", type=int, default=512, help="Maximum sequence length for LSTM")
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

    # Prepare data
    tokenizer = LSTMTokenizer(max_seq_length=args.max_seq_length)
    test_dataset, _, _, vocab_size = prepare_lstm_data(args.data_path,
                                    text_col=args.text_column,
                                    label_col=args.label_column,
                                    tokenizer=tokenizer,
                                    batch_size=args.batch_size,
                                    max_seq_length=args.max_seq_length, val_split=0.0, test_split=0.0, return_datasets=True)

    test_loader, _, _, vocab_size = prepare_lstm_data(args.data_path,
                                  text_col=args.text_column,
                                  label_col=args.label_column,
                                  tokenizer=tokenizer,
                                  batch_size=args.batch_size,
                                  max_seq_length=args.max_seq_length, val_split=0.0, test_split=0.0)
    
    # Load model
    model = DocumentBiLSTM(input_dim=tokenizer.vocab_size,
                           embedding_dim=128,
                           hidden_dim=64,
                           output_dim=args.num_classes)
    
    model.load_state_dict(torch.load(args.model_path))
    model = model.to(device)

    all_labels = np.array([], dtype=int)
    all_predictions = np.array([], dtype=int)
    
    # Inference
    batch_count = 0
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch['input_ids'].to(device)
            labels = batch['label'].to(device)
            all_labels = np.append(all_labels, labels.cpu().numpy())

            outputs = model(input_ids)
            predictions = torch.argmax(outputs, dim=1)
            all_predictions = np.append(all_predictions, predictions.cpu().numpy())

            if args.print_predictions:
                for i in range(len(predictions)):
                    print(f"Text: {test_dataset.get_text_(batch_count * args.batch_size + i)}, Prediction: {class_names[predictions[i]]}, True Label: {class_names[labels[i]]}")
                
            if args.inference_batch_limit > 0 and batch_count >= args.inference_batch_limit:
                break

            batch_count += 1

    # Print classification report
    # Calculate accuracy, F1 score, recall, and precision
    accuracy = sklearn.metrics.accuracy_score(all_labels, all_predictions)
    f1 = sklearn.metrics.f1_score(all_labels, all_predictions, average='weighted')
    precision = sklearn.metrics.precision_score(all_labels, all_predictions, average='weighted')
    recall = sklearn.metrics.recall_score(all_labels, all_predictions, average='weighted')

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
            
