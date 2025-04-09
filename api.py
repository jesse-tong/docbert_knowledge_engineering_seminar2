from model import DocBERT
from models.lstm_model import DocumentBiLSTM
from dataset import DataLoader, DocumentDataset
from utils.word_segmentation_vi import word_segmentation_vi
import numpy as np
from transformers import AutoTokenizer
import torch.nn.functional as F
import torch

args = {
    "bert_model": "vinai/phobert-base-v2", # Base BERT model name
    "model_path": "./vietnamese_hate_speech_detection_phobert/vinai_phobert-base-v2_finetuned.pth", # Change this if you have a fine-tuned model somewhere else
    "lstm_model_path": "./vietnamese_hate_speech_detection_phobert/distilled_lstm_model.pth", # Change this if you have a fine-tuned model somewhere else
    "max_seq_length": 250,
    "num_classes": 4, # As the fine tuned model has 4 classes per category
    "num_categories": 5, # As the fine tuned model has 5 categories
}

class_names = ["NORMAL", "CLEAN", "OFFENSIVE", "HATE"]

def load_model_bert():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = DocBERT(bert_model_name=args["bert_model"], num_classes=args["num_classes"], num_categories=args["num_categories"])
    model.load_state_dict(torch.load(args["model_path"], map_location=device))
    model = model.to(device)
    return model, device

def load_model_lstm():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(args["bert_model"])
    vocab_size = tokenizer.vocab_size
    model = DocumentBiLSTM(vocab_size=vocab_size,
                           embedding_dim=300,
                           hidden_dim=256,
                           n_layers=2,
                           output_dim=args["num_classes"] * args["num_categories"])
    model.load_state_dict(torch.load(args["lstm_model_path"], map_location=device)["model_state_dict"])
    model = model.to(device)
    return model, device

def inference(model, device, comments: str | list):
    if isinstance(comments, str):
        comments = [comments]
    elif not isinstance(comments, list):
        raise ValueError("comment must be a string or a list of strings")
    
    comments = np.array([word_segmentation_vi(comment) for comment in comments])
    data = DocumentDataset(texts=comments, labels=None, tokenizer_name=args["bert_model"], max_length=args["max_seq_length"])
    inference_loader = DataLoader(data, batch_size=comments.shape[0], shuffle=False)

    batch = next(iter(inference_loader))
    input_ids = batch['input_ids']
    attention_mask = batch['attention_mask']
    token_type_ids = batch['token_type_ids']

    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)
    token_type_ids = token_type_ids.to(device)

    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)
        if args["num_categories"] > 1:
            batch_size, total_classes = outputs.shape
            if total_classes % args["num_categories"] != 0:
                raise ValueError("Error: Number of total classes in the batch must of divisible by the number of categories.")

            classes_per_group = total_classes // args["num_categories"]
            # Group every classes_per_group values along dim=1
            reshaped = outputs.view(outputs.size(0), -1, classes_per_group)  # shape: (batch, self., classes_per_group)
            probs = F.softmax(reshaped, dim=1)
            # Argmax over each group of classes_per_group
            predictions = probs.argmax(dim=-1)
        else:
            predictions = torch.argmax(outputs, dim=-1)

    preds_array = predictions.cpu().numpy()
    result = []
    for i in range(preds_array.shape[0]):
        result.append(
        {
            "Bình luận": comments[i],
            "Cá nhân": class_names[ preds_array[i, 0] ],
            "Nhóm/tổ chức": class_names[ preds_array[i, 1] ],
            "Tôn giáo/tín ngưỡng": class_names[ preds_array[i, 2] ],
            "Chủng tộc/sắc tộc": class_names[ preds_array[i, 3] ],
            "Chính trị": class_names[ preds_array[i, 4] ],
        })
    return result

if __name__ == "__main__":
    
    model, device = load_model_bert()
    comments = [
        "Để avata bít ngay là ngu hơn chó",
        "Hàn Quốc chửi dân Đông Lào và đây là hậu quả",
        "Nguyễn Thuận =)) tư tưởng rừng rú gì vậy",
        "@công danh nguyen thể chế chính trị khác hẳn tư tưởng xã hội nhé. Con cờ hó china liên quan cmn gì?"
    ]
    predictions = inference(model, device, comments)
    print("BERT Predictions:")
    print(predictions)

    lstm_model, device = load_model_lstm()
    lstm_predictions = inference(lstm_model, device, comments)
    print("LSTM Predictions:")
    print(lstm_predictions)