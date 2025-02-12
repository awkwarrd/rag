from typing import TypedDict
from typing_extensions import Annotated, Sequence
from dotenv import load_dotenv

from langchain.prompts import ChatPromptTemplate
from langchain.tools.retriever import create_retriever_tool

from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import BaseMessage
from langchain_core.prompts.chat import MessagesPlaceholder

from langchain_ollama import ChatOllama

from langchain_pinecone import PineconeVectorStore

from langchain_community.tools.tavily_search import TavilySearchResults

from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

from agents import EMBEDDINGS, SYSTEM_PROMPT, generate_output

import streamlit as st

load_dotenv()

class State(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages] 

vectorstore = PineconeVectorStore(index_name="miami", embedding=EMBEDDINGS)
retriever = vectorstore.as_retriever(search_type="similarity_score_threshold", search_kwargs={"k": 3, "score_threshold": 0.8})
retrieve_tool = create_retriever_tool(
    retriever=retriever,
    name="retrieve_from_pinecone",
    description="Search and retrieve data about Hotline Miami Series lore from Youtube videos and Fandom Wiki"
)

tool = TavilySearchResults(max_results=2)
tools = [tool]

llm = ChatOllama(model="llama3.2")

def chatbot(state:State):
    return {"messages" : [llm.invoke(state["messages"])] }


def generate(state: State):
    message = state["messages"]
    question = message[-2].content
    docs_content = message[-1].content

    prompt = create_prompt()
    rag_chain = prompt | llm | StrOutputParser()

    response = rag_chain.invoke({"context" : docs_content, "question" : question, "messages" : message})
    return {"messages" : [response]}

def retrieve(state: State):
    retriever = vectorstore.as_retriever(search_type="similarity_score_threshold", search_kwargs={"k": 3, "score_threshold": 0.8})

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
    

    graph_builder.add_conditional_edges(START, route_tools, {"search" : "search", "retrieve" : "retrieve"})
    graph_builder.add_edge("retrieve", "generate")
    graph_builder.add_edge("search", "generate")
    graph_builder.add_edge("generate", END)
    
    graph = graph_builder.compile(checkpointer=memory)

    return graph


def stream_graph(user_input:str, graph, config):
    for event in graph.stream({"messages": [{"role": "user", "content": user_input}]}, config):
        for value in event.values():
            print("Assistant:", value["messages"][-1].content)


def debug_stream(inputs, config):
    for output in graph.stream(inputs, config=config):
        for key, value in output.items():
            print(f"Output from node '{key}':")
            print("------------------------------------------------------------------------------------------------------------------------------------------------------")
            print(value)
        print("\n------------------------------------------------------------------------------------------------------------------------------------------------------\n")


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