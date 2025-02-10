from agents import create_chain, generate_output, create_prompt
from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage
import streamlit as st

load_dotenv()

rag = create_chain(create_prompt)

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message.type):
        st.markdown(message.content)


if prompt := st.chat_input("Ask any question about Hotline Miami Series:"):
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append(HumanMessage(prompt))

    with st.chat_message("ai"):
        response = rag.invoke({"input" : prompt, "messages": st.session_state.messages})["answer"] 
        st.write_stream(generate_output(response))
    st.session_state.messages.append(AIMessage(response))