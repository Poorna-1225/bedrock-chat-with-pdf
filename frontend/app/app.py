import streamlit as st
import requests
import os
from dotenv import load_dotenv

load_dotenv()
url = os.getenv("BACKEND_API_URL")

# FastAPI backend URL
BACKEND_URL = url

st.title("PDF Q&A with Retrieval-Augmented Generation")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Hi! Upload a PDF and ask me anything about it."}]

# --- Sidebar ---
with st.sidebar:
    uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")
    if uploaded_file and "pdf_processed"not in st.session_state:
        # Save the uploaded file to a temporary file
        with open("temp.pdf", "wb") as f:
            f.write(uploaded_file.getbuffer())
        st.write(f"Uploaded File: {uploaded_file.name}")

        # Send the file to the FastAPI backend for processing
        with open('temp.pdf','rb')as f:
            files = {'file': f}
            with st.spinner("Processing PDF..."):
                
                response = requests.post(f"{BACKEND_URL}/upload-and-process", files=files)
                if response.status_code == 200:
                    st.sidebar.success("PDF processed and index created successfully!")
                    st.session_state.pdf_processed = True
                else:
                    st.sidebar.error(f"Error: {response.json().get('detail', 'Unknown error')}")

        # Clean up the temporary file
        os.remove("temp.pdf")

# --- Main Chat Area ---
chat_container = st.container()
with chat_container:
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

# User query input (at the bottom of the main area)
user_input = st.chat_input("Enter your question here...")

# Force the chat container to expand to fill available space (optional but recommended)
st.markdown("<style>div[data-testid='column'] { flex-grow: 1; }</style>", unsafe_allow_html=True)

if user_input:
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": user_input})

    # Display user message
    with chat_container:
        with st.chat_message("user"):
            st.markdown(user_input)

    # Send the query to the FastAPI backend
    with st.spinner('Generating response...'):
        response = requests.post(f"{BACKEND_URL}/query", json={"query": user_input})
        if response.status_code == 200:
            answer = response.json().get( "response","No response generated.")
        else:
            answer = f"Error: {response.json().get('detail', 'Unknown error')}"

    # Add bot response to chat history
    st.session_state.messages.append({"role": "assistant", "content": answer})

    # Display bot response
    with chat_container:
        with st.chat_message("assistant"):
            st.markdown(answer)