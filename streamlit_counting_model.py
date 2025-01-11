import streamlit as st
import numpy as np
import joblib
import re
from transformers import AutoTokenizer, AutoModel
import torch
import scipy.sparse as sp

# ==========================
# 1. Load Model dan IndoBERT
# ==========================
model_file = 'lightgbm_model.pkl'
scaler_file = 'scaler.pkl'

model = joblib.load(model_file)
scaler = joblib.load(scaler_file)

# Load pre-trained IndoBERT model dan tokenizer
model_name = "indobenchmark/indobert-base-p2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
indobert_model = AutoModel.from_pretrained(model_name)

# Domain mapping
domain_mapping = {
    'news.detik.com': 7,
    'detik.com': 0,
    'hot.detik.com': 5,
    'wolipop.detik.com': 11,
    'health.detik.com': 4,
    'finance.detik.com': 1,
    'sport.detik.com': 9,
    'inet.detik.com': 6,
    'food.detik.com': 2,
    'travel.detik.com': 10,
    'oto.detik.com': 8,
    'haibunda.com': 3,
}

# ==========================
# 2. Fungsi Preprocessing dan Encoding
# ========================== 
def preprocess_text(text):
    text = re.sub(r'http\S+|https\S+|www\S+|ftp\S+', '', text)  # Remove URLs
    text = re.sub(r'[^\w\s]', '', text.lower())  # Remove punctuation and lowercase
    return text

def encode_text_with_indobert(texts):
    """
    Fungsi untuk menghasilkan embedding menggunakan IndoBERT.
    """
    tokens = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=128,
        return_tensors="pt"
    )
    with torch.no_grad():
        outputs = indobert_model(**tokens)
    return outputs.last_hidden_state[:, 0, :].cpu().numpy()  # CLS token

# ==========================
# 3. Streamlit Interface
# ==========================
st.title("Prediksi Impression Pembaca Postingan Berita Detik.com")

# Input text
user_text = st.text_area("Masukkan Teks Postingan X")
retweets = st.number_input("Masukkan Jumlah Retweets", min_value=0, value=0, step=1)
domain = st.selectbox("Pilih Domain", options=list(domain_mapping.keys()))

if st.button("Prediksi"):
    if user_text.strip():
        # Preprocess text
        processed_text = preprocess_text(user_text)
        text_length = len(processed_text)

        # Convert text to IndoBERT embeddings
        st.write("Menghasilkan embedding IndoBERT...")
        text_embedding = encode_text_with_indobert([processed_text])
        text_sparse = sp.csr_matrix(text_embedding)

        # Encode domain
        encoded_domain = sp.csr_matrix([[domain_mapping[domain]]])

        # Retweets sparse matrix
        retweets_sparse = sp.csr_matrix([[retweets]])

        # Length sparse matrix
        length_sparse = sp.csr_matrix([[text_length]])

        # Combine features
        input_features = sp.hstack([text_sparse, encoded_domain, retweets_sparse, length_sparse])
        st.write("Dimensi input_features:", input_features.shape)
        st.write("Scaler di-fit pada dimensi fitur:", scaler.n_features_in_)

        # Scale features
        scaled_features = scaler.transform(input_features)

        # Predict
        prediction = model.predict(scaled_features)
        st.success(f"Perkiraan pembaca sebanyak: {prediction[0]:,.0f} orang")
    else:
        st.warning("Tolong masukkan teks untuk prediksi.")
