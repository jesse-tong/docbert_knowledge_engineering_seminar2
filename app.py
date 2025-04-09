import streamlit as st
from api import load_model_bert, load_model_lstm, inference
import pandas as pd
import git

# Set up the Streamlit app
def app():
    st.set_page_config(layout="wide")
    st.title("Phân tích ngôn từ thù địch, phân biệt sử dụng PhoBERT và LSTM")
    
    # Show loading progress bar
    # Load models
    @st.cache_resource
    def load_models():
        loading_model_bar = st.progress(0, "Nạp các mô hình...")
        # Load BERT model
        bert_model, bert_device = load_model_bert()
        loading_model_bar.progress(50, "Mô hình PhoBERT đã được nạp.")
        # Load LSTM model
        lstm_model, lstm_device = load_model_lstm()
        loading_model_bar.progress(100, "Mô hình LSTM đã được nạp.")  # Complete loading progress
        loading_model_bar.empty()
        return bert_model, bert_device, lstm_model, lstm_device

    bert_model, bert_device, lstm_model, lstm_device = load_models()
    
    # User input
    user_input = st.text_area("Nhập các bình luận để phân tích ngôn từ thù địch, phân biệt (xuống dòng cho từng bình luận):")

    if st.button("Phân tích"):
        if user_input:
            # Preprocess input
            comments = user_input.splitlines()

            # Inference with BERT
            classification_bar = st.progress(0, "Đang phân tích với PhoBERT...")
            bert_predictions = inference(bert_model, bert_device, comments)
            st.write("Phân loại của PhoBERT:")
            st.table(pd.DataFrame(bert_predictions))

            classification_bar.progress(50, "Đang phân tích với LSTM...")

            # Inference with LSTM
            lstm_predictions = inference(lstm_model, lstm_device, comments)
            st.write("Phân loại của LSTM:")
            classification_bar.progress(100, "Phân tích hoàn tất!")
            classification_bar.empty()
            st.table(pd.DataFrame(lstm_predictions))
        else:
            st.warning("Hãy nhập một vài bình luận.")

if __name__ == "__main__":
    # Clone from https://huggingface.co/jesse-tong/vietnamese_hate_speech_detection_phobert
    # if the directory named ./vietnamese_hate_speech_detection_phobert does not exist
    # If you want to train the model instead of using the pre-trained model, please refer to the example_uses.md
    repo_url = "https://huggingface.co/jesse-tong/vietnamese_hate_speech_detection_phobert"
    repo_dir = "./vietnamese_hate_speech_detection_phobert"
    if not git.Repo(repo_dir).git_dir.exists():
        git.Repo.clone_from(repo_url, repo_dir)
    app()