import streamlit as st
import json
import numpy as np
import pickle
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input, Embedding, GlobalAveragePooling1D
from tensorflow.keras.preprocessing.sequence import pad_sequences
import tensorflow.keras
from tensorflow.keras.preprocessing.text import Tokenizer

# =====================
# Load files
# =====================
with open("intents.json") as file:
    data = json.load(file)

with open("tokenizer.pickle", "rb") as f:
    tokenizer = pickle.load(f)

with open("label_encoder.pickle", "rb") as f:
    lbl_encoder = pickle.load(f)

# =====================
# Parameters (must match training)
# =====================
vocab_size = 1000
embedding_dim = 16
max_len = 20
num_classes = len(lbl_encoder.classes_)

# =====================
# Build model architecture
# =====================
def build_model():
    model = Sequential()
    model.add(Input(shape=(max_len,)))
    model.add(Embedding(vocab_size, embedding_dim))
    model.add(GlobalAveragePooling1D())
    model.add(Dense(16, activation='relu'))
    model.add(Dense(16, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))
    return model

model = build_model()

# =====================
# Load weights
# =====================
model.load_weights("chatbot_model.weights.h5")

# =====================
# Prediction function
# =====================
def predict_class(text):
    seq = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(seq, truncating='post', maxlen=max_len)

    result = model.predict(padded, verbose=0)
    tag = lbl_encoder.inverse_transform([np.argmax(result)])[0]

    for intent in data["intents"]:
        if intent["tag"] == tag:
            return np.random.choice(intent["responses"])

    return "Sorry, I didn't understand that."

# =====================
# Streamlit UI
# =====================
st.title("💬 AI Chatbot")
st.write("Ask me anything!")

user_input = st.text_input("You:")

if user_input:
    response = predict_class(user_input)
    st.success(response)
