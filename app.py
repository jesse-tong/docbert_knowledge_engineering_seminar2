import streamlit as st
from api import load_model_bert, load_model_lstm, inference
import pandas as pd

# Set up the Streamlit app
def app():
    st.title("Phân tích ngôn từ thù địch, phân biệt sử dụng PhoBERT và LSTM")

      # Show loading progress bar
    # Load models
    @st.cache_resource
    def load_models():
        st.progress(0, "Nạp các mô hình...")
        # Load BERT model
        bert_model, bert_device = load_model_bert()
        st.progress(50, "Mô hình PhoBERT đã được nạp.")
        # Load LSTM model
        lstm_model, lstm_device = load_model_lstm()
        st.progress(100, "Mô hình LSTM đã được nạp.")  # Complete loading progress
        return bert_model, bert_device, lstm_model, lstm_device

    bert_model, bert_device, lstm_model, lstm_device = load_models()
    
    # User input
    user_input = st.text_area("Nhập các bình luận để phân tích ngôn từ thù địch, phân biệt (xuống dòng cho từng bình luận):")

    if st.button("Phân tích"):
        if user_input:
            # Preprocess input
            comments = user_input.splitlines()

            # Inference with BERT
            st.progress(0, "Đang phân tích với PhoBERT...")
            bert_predictions = inference(bert_model, bert_device, comments)
            st.write("BERT Predictions:")
            st.dataframe(pd.DataFrame(bert_predictions))

            st.progress(50, "Đang phân tích với LSTM...")

            # Inference with LSTM
            lstm_predictions = inference(lstm_model, lstm_device, comments)
            st.write("LSTM Predictions:")
            st.progress(100, "Phân tích hoàn tất!")
            st.dataframe(pd.DataFrame(lstm_predictions))
        else:
            st.warning("Hãy nhập một vài bình luận.")

if __name__ == "__main__":
    app()