import streamlit as st
from intent_chatbot import predict_intent  

# Set page title and icon
st.set_page_config(page_title="Intent-Based Chatbot", page_icon="🤖")

# App Title and Description
st.title("🤖 Intent-Based Chatbot")
st.write("Welcome! Ask me anything, and I'll try to understand your intent and respond accordingly.")

# Initialize session state to store chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# to display chat messages in a chat-like format
def display_chat():
    for message in st.session_state.chat_history:
        if message["role"] == "user":
            st.markdown(f'<div style="text-align: right; color: blue; margin: 10px;">YOU: {message["text"]}</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div style="text-align: left; color: green; margin: 10px;">CHATBOT: {message["text"]}</div>', unsafe_allow_html=True)

# Display chat history
display_chat()

# Creating  form for the user's message input and submit button
with st.form("message_form"):
    user_input = st.text_input("Type your message here and press Enter:", key="user_input", placeholder="Ask me anything...")
    # submit button
    submit_button = st.form_submit_button("Submit")

# When the user sends a message
if submit_button and user_input:
    # the chatbot's response
    response = predict_intent(user_input)
    
    # Add the user's message and chatbot's response to the chat history
    st.session_state.chat_history.append({"role": "user", "text": user_input})
    st.session_state.chat_history.append({"role": "chatbot", "text": response})
    # Clear the input box by updating the key to an empty string
    st.rerun()

#"Clear Chat" button to reset the chat history
if st.button("Clear Chat"):
    st.session_state.chat_history = []
    st.rerun()

#styling to make the app more attractive
st.markdown("""
<style>
    .stTextInput input {
        border-radius: 10px;
        padding: 10px;
    }
    .stButton button {
        background-color: #4CAF50;
        color: white;
        border-radius: 10px;
        padding: 10px 20px;
        border: none;
        cursor: pointer;
    }
    .stButton button:hover {
        background-color: #45a049;
    }
</style>
""", unsafe_allow_html=True)