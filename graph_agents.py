from typing import TypedDict
from typing_extensions import Annotated, Sequence

from langchain_core.messages import HumanMessage, AIMessage, RemoveMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import BaseMessage
from langchain_core.prompts.chat import MessagesPlaceholder, ChatPromptTemplate

from langchain_ollama import ChatOllama

from langchain_pinecone import PineconeVectorStore

from langchain_community.tools.tavily_search import TavilySearchResults

from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

import spacy
from fastcoref import spacy_component

from agents import EMBEDDINGS, SYSTEM_PROMPT, generate_output
import streamlit as st

LLM = ChatOllama(model="llama3.2", temperature=0.1)
VECTORSTORE = PineconeVectorStore(index_name="miami", embedding=EMBEDDINGS)
NLP = spacy.load("en_core_web_lg")
NLP.add_pipe("fastcoref")


class State(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages] 

def resolve_coreference(state:State):
    conversation = " $ ".join([x.content for x in state["messages"][::2]])
    doc = NLP(conversation, component_cfg={"fastcoref": {'resolve_text': True}})

    resolved = doc._.resolved_text
    deleted_message = [RemoveMessage(state["messages"][-1].id)]
    return {"messages" : deleted_message + [resolved.split(" $ ")[-1]]}


def generate(state: State):
    message = state["messages"]
    question = message[-2].content
    docs_content = message[-1].content

    prompt = create_prompt()
    rag_chain = prompt | LLM | StrOutputParser()

    response = rag_chain.invoke({"context" : docs_content, "question" : question, "messages" : message})
    return {"messages" : [response]}

def retrieve(state: State):
    retriever = VECTORSTORE.as_retriever(search_type="similarity_score_threshold", search_kwargs={"k": 3, "score_threshold": 0.8})

    query = state["messages"][-1].content
    retrieved_docs = retriever.invoke(query)

    docs_content = " ".join([doc.page_content for doc in retrieved_docs])
    
    
    return {"messages": [{"role": "user", "content": docs_content}]}

def search(state: State):

    query = state["messages"][-1].content
    searcher = TavilySearchResults(max_results=2)
    search = searcher.invoke(query)

    return {"messages" : [{"role" : "user", "content" : str(search)}]}


def route_tools(state:State):
    if isinstance(state, list):
        message = state[-1]
    elif messages := state.get("messages", []):
        message = messages[-1]
    else:
        raise ValueError(f"No messages found in input state to tool_edge: {state}")
    
    if "/search" in message.content.lower().split():
        return "search"
    return "retrieve"


def create_prompt():
    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("human", "{question}"),
        MessagesPlaceholder(variable_name="messages")
])
    return prompt


def create_graph():
    memory = MemorySaver()
    graph_builder = StateGraph(State)
    
    graph_builder.add_node("retrieve", retrieve)
    graph_builder.add_node("generate", generate)
    graph_builder.add_node("search", search)
    graph_builder.add_node("resolve_coref", resolve_coreference)
    

    graph_builder.add_edge(START, "resolve_coref")
    graph_builder.add_conditional_edges("resolve_coref", route_tools, {"search" : "search", "retrieve" : "retrieve"})
    graph_builder.add_edge("retrieve", "generate")
    graph_builder.add_edge("search", "generate")
    graph_builder.add_edge("generate", END)
    
    graph = graph_builder.compile(checkpointer=memory)

    return graph


def stream_graph(user_input:str, graph, config):
    for event in graph.stream({"messages": [{"role": "user", "content": user_input}]}, config):
        for value in event.values():
            print("Assistant:", value["messages"][-1].content)


def debug_stream(inputs, config, app):
    for output in app.stream(inputs, config=config):
        for key, value in output.items():
            print(f"Output from node '{key}':")
            print("----------------------------------------------------------------------------------------------------------------------------------------------")
            print(value)
        print("\n-----------------------------------------------------------------------------------------------------------------------------------------------\n")


def start_chat():
    
    graph = create_graph()
    config = {"configurable" : {"thread_id" : "1"}}

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message.type):
            st.markdown(message.content)

    if prompt := st.chat_input("Ask any question or just start conversation"):
        with st.chat_message("user"):
            st.markdown(prompt)
        st.session_state.messages.append(HumanMessage(prompt))

        with st.chat_message("ai"):
            response = graph.invoke({"messages" : st.session_state.messages}, config=config)["messages"][-1].content
            print(prompt)
            print(response)
            st.write_stream(generate_output(response))
        st.session_state.messages.append(AIMessage(response))