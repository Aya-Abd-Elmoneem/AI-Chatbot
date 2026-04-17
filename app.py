import streamlit as st
import json
import numpy as np

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input, Embedding, GlobalAveragePooling1D
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import tokenizer_from_json

# =====================
# Load intents
# =====================
with open("intents.json") as file:
    data = json.load(file)

# =====================
# Load tokenizer (JSON instead of pickle)
# =====================
with open("tokenizer.json") as f:
    tokenizer = tokenizer_from_json(f.read())

# =====================
# Load label encoder
# =====================
with open("labels.json") as f:
    labels = json.load(f)

# =====================
# Model parameters (must match training)
# =====================
vocab_size = 1000
embedding_dim = 16
max_len = 20
num_classes = len(labels)

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
    tag = labels[np.argmax(result)]

    for intent in data["intents"]:
        if intent["tag"] == tag:
            return np.random.choice(intent["responses"])

    return "Sorry, I didn't understand that."

# =====================
# Streamlit UI
# =====================
st.set_page_config(page_title="AI Chatbot", page_icon="💬")

st.title("💬 AI Chatbot")
st.write("Ask me anything!")

user_input = st.text_input("You:")

if user_input:
    response = predict_class(user_input)
    st.success(response)
