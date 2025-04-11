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

def inference(model, device, comments: str | list, threshold: float = 0.55):
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

            # Keep only the probs that are above the threshold (to prevent false positive), else set it to 0 (NORMAL, in this case unconclusive)
            probs = torch.where(probs > threshold, probs, 0.0)
            print("Probabilities: ", probs)
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
    '''comments = [
        "Em ăn hoành thánh sáng bị khó chịu mắc ói quá bỏ ăn trưa luôn. Các thím thường hay uống gì cho đỡ vậy? Em tính làm gói gừng pha uống",
        "Quan trọng là năm nay có tham gia những lễ hội có tính chất, quy mô và bối cảnh y hệt vậy không? Chứ tôi nói thật, dù ở bất cứ đâu mà tập trung đông đến mức không tiến không lùi như này được thì đều nguy hiểm. Khoan nói về giẫm đạp, chỉ riêng việc có sự cố đột xuất xảy ra thì chuyện cấp cứu nó sẽ vô cùng khó khăn và mất rất nhiều thời gian. Bởi vậy, tôi từ chối tham gia tất cả lễ hội nơi mà số người vượt tải đến mức không thể nhúc nhích như thế này.",
        "Còn phải tốn hơn nữa mới được",
        "Mình k có ý kích dục fen nhé :v Có sao kể vậy thôi.",
        "Này là lúc trước khi gặp P hả bác? Em thắc mắc là bác có thể thẳng thừng chặn C - người bác yêu như vậy à?",
        "Thì mượt hơn là đúng thôi. Mới phát triển thì không có nhiều tính năng, không có nhiều app thì chả mượt",
    ]'''
    comments = [
        "đúng là vozer, nhiều thằng sống ngu và ích kỷ vcl, nếu như người yêu nó cần 1 trái thận, lúc đó bản thân suy nghĩ tính toán thì ok, này chạy xe có 40km mà tính toán chi ly, mua cái váy mà mặc đi",
        "Khác mẹ gì tàu khựa, bơm tiền cho đám NGO woke đi biểu tình phá lại bọn tây lông thôi. Chó chê mèo lắm lông. À mà acc Emma Roberts bị ban rồi à mày",
        "đùa, cái shop thế mà cũng bảo chính hãng, vả vỡ alo nó đi. ra trung tâm thương mại, hay cửa hàng chính hãng mà mua.",
        "qua thớt này của nó thì 90% là xiaolol rùi",
        "thằng này chuyên đăng bài để hả hê, khóa mõm nó đi mod",
        "Đm nhẫm vào đuổi con bò đỏ này nó giãy nảy cắn người kinh thật @@ Tao có hay ko liên quan lol gì mà mày có vẻ cay cú vkl nhỉ, chắc gato với tao hả ))",
        "Sao thế óc chó, bị chửi cho ngu người rồi à =]] thứ ngu học chả biết mẹ gì vào sủa như đúng rồi =]]",
    ]
    predictions = inference(model, device, comments)
    print("BERT Predictions:")
    print(predictions)

    lstm_model, device = load_model_lstm()
    lstm_predictions = inference(lstm_model, device, comments)
    print("LSTM Predictions:")
    print(lstm_predictions)