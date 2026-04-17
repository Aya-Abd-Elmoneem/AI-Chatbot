import streamlit as st

st.title("🤖 My Chatbot")
st.write("Hello from Streamlit!")

user_input = st.text_input("You:")

if st.button("Send"):
    if user_input:
        response = chatbot_response(user_input)
        st.write("Bot:", response)
