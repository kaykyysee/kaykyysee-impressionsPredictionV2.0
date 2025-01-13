import streamlit as st
import numpy as np
import pandas as pd
import joblib
import re
from transformers import AutoTokenizer, AutoModel
import torch
import scipy.sparse as sp
import nltk
from nltk.tokenize import word_tokenize
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
import string
import pickle

# Unduh dataset punkt jika belum tersedia
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

# ==========================
# 1. Fungsi Caching untuk Load Model dan Tokenizer
# ==========================
@st.cache_resource
def load_model_and_tokenizer():
    model_file = 'catboost_model.pkl'
    scaler_file = 'scaler.pkl'

    # Load model dan scaler
    model = joblib.load(model_file)
    scaler = joblib.load(scaler_file)

    # Load pre-trained IndoBERT model dan tokenizer
    model_name = "indobenchmark/indobert-base-p2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    indobert_model = AutoModel.from_pretrained(model_name, device_map=None).to("cpu")

    return model, scaler, tokenizer, indobert_model

# Load model dan tokenizer sekali saja
model, scaler, tokenizer, indobert_model = load_model_and_tokenizer()

# ==========================
# 2. Fungsi Preprocessing dan Encoding
# ==========================
@st.cache_data
def preprocess_text(text):
    """
    Preprocessing teks untuk menghapus URL, tanda baca, dan huruf besar.
    """
    text = re.sub(r'http\S+|https\S+|www\S+|ftp\S+', '', text)  # Hapus URL
    text = re.sub(r'\b[a-zA-Z0-9]+\.com\S*', '', text)  # Hapus domain seperti example.com/link
    text = text.translate(str.maketrans('', '', string.punctuation)).lower()
    text = re.sub(r'\s+', ' ', text).strip()
    return text

@st.cache_data
def clean_text_id(text):
    """
    Fungsi untuk membersihkan teks, menghapus stop words, dan melakukan stemming.
    """
    tokens = word_tokenize(text)

    # Inisialisasi stop words dan stemmer
    stop_words_id = set(StopWordRemoverFactory().get_stop_words())
    stemmer = StemmerFactory().create_stemmer()

    # Hapus stop words dan lakukan stemming
    tokens = [stemmer.stem(word) for word in tokens if word not in stop_words_id]

    return ' '.join(tokens)

@st.cache_data
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

@st.cache_data
def one_hot_encode_domain(domain, unique_domains):
    """
    One-hot encode domain menjadi sparse matrix.
    """
    one_hot = pd.DataFrame([domain], columns=["domain"])
    one_hot_encoded = pd.get_dummies(one_hot, columns=["domain"], prefix="domain")

    # Pastikan semua kategori dari unique_domains muncul
    for col in unique_domains:
        if col not in one_hot_encoded:
            one_hot_encoded[col] = 0

    # Pastikan tipe data numerik
    one_hot_encoded = one_hot_encoded.astype(float)

    # Validasi jumlah fitur
    if len(one_hot_encoded.columns) != len(unique_domains):
        raise ValueError(
            f"Jumlah fitur encoding ({len(one_hot_encoded.columns)}) tidak sesuai dengan jumlah yang diharapkan ({len(unique_domains)})."
        )

    return sp.csr_matrix(one_hot_encoded.values)

@st.cache_resource
def load_domains():
    """
    Load daftar domain dari file domains.pkl.
    """
    with open("domains.pkl", "rb") as f:
        domains = pickle.load(f)
    return domains

# Muat daftar domain unik
unique_domains = load_domains()

# ==========================
# 3. Streamlit Interface
# ==========================
st.title("Prediksi Jumlah Tayangan Postingan Berita Detik.com 📰")

# Input text
user_text = st.text_area("Masukkan Teks Postingan X", height=150, placeholder="Tulis atau paste teks di sini...")
retweets = st.number_input("Masukkan Jumlah Retweets", min_value=0, value=0, step=1)
unique_domains = [f"domain_{x}" for x in [
    "news.detik.com", "detik.com", "hot.detik.com", "wolipop.detik.com",
    "health.detik.com", "finance.detik.com", "sport.detik.com", "inet.detik.com",
    "food.detik.com", "travel.detik.com", "oto.detik.com", "haibunda.com"
]]
domain = st.selectbox("Pilih Domain", options=[d.split("_")[1] for d in unique_domains])

if st.button("Prediksi"):
    if user_text.strip():
        # Preprocess text
        processed_text = preprocess_text(user_text)
        cleaned_text = clean_text_id(processed_text)
        text_length = len(cleaned_text.split())

        # Convert text to IndoBERT embeddings
        text_embedding = encode_text_with_indobert([cleaned_text])
        text_sparse = sp.csr_matrix(text_embedding)

        # One-hot encode domain
        encoded_domain = one_hot_encode_domain(f"domain_{domain}", unique_domains)

        # Retweets sparse matrix
        retweets_sparse = sp.csr_matrix([[retweets]])

        # Length sparse matrix
        length_sparse = sp.csr_matrix([[text_length]])

        # Combine features
        input_features = sp.hstack([text_sparse, encoded_domain, retweets_sparse, length_sparse])
        
        # Validasi jumlah fitur
        if input_features.shape[1] != scaler.n_features_in_:
            st.error(
                f"Jumlah fitur input ({input_features.shape[1]}) tidak sesuai dengan jumlah yang diharapkan ({scaler.n_features_in_})."
            )
        else:
            # Scale features
            scaled_features = scaler.transform(input_features)
        
            # Predict
            prediction = model.predict(scaled_features)
            st.success(f"Diperkirakan sebanyak: {prediction[0]:,.0f} penayangan akan dicapai dalam 1 minggu")

    else:
        st.warning("Tolong masukkan teks untuk prediksi.")
