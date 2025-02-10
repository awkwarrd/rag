from agents import create_chain, generate_output
from dotenv import load_dotenv
import streamlit as st

load_dotenv()

rag = create_chain()

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


if prompt := st.chat_input("Ask any question about Hotline Miami Series:"):
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("ai"):
        response = rag.invoke({"input" : prompt})["answer"] 
        st.write_stream(generate_output(response))
    st.session_state.messages.append({"role": "assistant", "content": response})