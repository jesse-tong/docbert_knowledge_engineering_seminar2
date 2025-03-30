from dataset_lstm import prepare_lstm_data, LSTMTokenizer, LSTMDataset
from models.lstm_model import DocumentBiLSTM
from sklearn import metrics
import torch, random
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Document Classification with LSTM")
    parser.add_argument("--data_path", type=str, required=True, help="Path to the dataset")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the trained model")
    parser.add_argument("--tokenizer_path", type=str, required=True, help="Path to the tokenizer")
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
    
    tokenizer = LSTMTokenizer(max_seq_length=args.max_seq_length)
    tokenizer.from_json(args.tokenizer_path)

    # Prepare data
    _, _, test_dataset, vocab_size = prepare_lstm_data(args.data_path,
                                    text_col=args.text_column,
                                    label_col=args.label_column,
                                    batch_size=args.batch_size,
                                    max_seq_length=args.max_seq_length, 
                                    val_split=0.0, test_split=1.0, 
                                    tokenizer=tokenizer, return_datasets=True,
                                    seed=random.randint(0, 10000))

    test_loader = DataLoader(test_dataset, batch_size=args.batch_size)
    
    # Load model
    model = DocumentBiLSTM(vocab_size=tokenizer.vocab_size,
                           embedding_dim=args.embedding_dim,
                           hidden_dim=args.hidden_dim,
                           n_layers=args.num_layers,
                           output_dim=args.num_classes)

    if 'model_state_dict' in model_state:
        model.load_state_dict(model_state['model_state_dict'])
    else:
        model.load_state_dict(model_state)

    model = model.to(device)

    all_labels = np.array([], dtype=int)
    all_predictions = np.array([], dtype=int)
    
    # Add this after model loading and before inference
    
    # Debug: Print model information
    print(f"Model loaded with vocab_size={tokenizer.vocab_size}")
    print(f"Model state contains keys: {model_state.keys()}")
    if 'config' in model_state:
        print(f"Model config: {model_state['config']}")

    # Create a verification step to check class alignment
    print("Class mapping verification:")
    print(f"Classes provided: {class_names}")
    print("First 5 examples with predictions:")

    # Check first 5 examples
    verify_loader = DataLoader(test_dataset, batch_size=5)
    batch = next(iter(verify_loader))
    with torch.no_grad():
        input_ids = batch['input_ids'].to(device)
        labels = batch['label'].to(device)
        
        outputs = model(input_ids)
        probs = F.softmax(outputs, dim=1)
        predictions = torch.argmax(probs, dim=1)
        
        for i in range(len(input_ids)):
            text = test_dataset.get_text_(i)
            true_label = labels[i].item()
            pred_label = predictions[i].item()
            confidence = probs[i][pred_label].item()
            
            print(f"Example {i}:")
            print(f"  Text: {text['text']}...")
            print(f"  True label index: {true_label}, Expected class: {class_names[true_label]}")
            print(f"  Predicted index: {pred_label}, Predicted class: {class_names[pred_label]}")
            print(f"  Confidence: {confidence:.2%}")
            print(f"  All probabilities: {probs[i].cpu().numpy()}")
            print("---")

    # Inference
    batch_count = 0
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch['input_ids'].to(device)
            labels = batch['label'].to(device)
            all_labels = np.append(all_labels, labels.cpu().numpy())

            outputs = model(input_ids)
            probs = F.softmax(outputs, dim=1)
            predictions = torch.argmax(probs, dim=1)
            all_predictions = np.append(all_predictions, predictions.cpu().numpy())

            if args.print_predictions:
                for i in range(len(predictions)):
                    print(f"Text: {test_dataset.get_text_(batch_count * args.batch_size + i)}, Prediction: {class_names[predictions[i]]}, True Label: {class_names[labels[i]]}")
                
            if args.inference_batch_limit > 0 and batch_count >= args.inference_batch_limit:
                break

            batch_count += 1

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
            
