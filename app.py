import streamlit as st
import json
import numpy as np
import pickle
from tensorflow import keras
from tensorflow.keras.preprocessing.sequence import pad_sequences

# load data
with open("intents.json") as file:
    data = json.load(file)

model = keras.models.load_model('chat_model.keras')

with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

with open('label_encoder.pickle', 'rb') as enc:
    lbl_encoder = pickle.load(enc)

max_len = 20

st.title("🤖 AI Chatbot")

if "messages" not in st.session_state:
    st.session_state.messages = []

user_input = st.text_input("You:")

if user_input:
    st.session_state.messages.append(("You", user_input))

    sequence = tokenizer.texts_to_sequences([user_input])
    padded = pad_sequences(sequence, truncating='post', maxlen=max_len)

    result = model.predict(padded)
    tag = lbl_encoder.inverse_transform([np.argmax(result)])[0]

    for intent in data['intents']:
        if intent['tag'] == tag:
            response = np.random.choice(intent['responses'])

    st.session_state.messages.append(("Bot", response))

for sender, msg in st.session_state.messages:
    st.write(f"**{sender}:** {msg}")
