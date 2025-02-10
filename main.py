from langchain_community.document_loaders import YoutubeLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_ollama.chat_models import ChatOllama
from langchain_ollama.embeddings import OllamaEmbeddings

from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from langchain import hub

from dotenv import load_dotenv

import streamlit as st

import requests
from bs4 import BeautifulSoup
import re
import os

load_dotenv()

embeddings = OllamaEmbeddings(model="nomic-embed-text")

def parse_video_and_store(url):

    loader = YoutubeLoader(url)
    transcription = loader.load()

    with open("transcription.txt", "w") as f:
        f.write(str(transcription))

    print("File was created successfully!")
    

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    docs = text_splitter.split_documents(transcription)

    pc = Pinecone()

    if "miami" not in pc.list_indexes().names():
        pc.create_index("miami", dimension=768, spec=ServerlessSpec("aws", "us-east-1"))

    PineconeVectorStore.from_documents(documents=docs, embedding=embeddings, index_name="miami")
    return 


def create_chain():
    vectorstore = PineconeVectorStore(index_name="miami", embedding=embeddings)
    retriever = vectorstore.as_retriever()

    llm = ChatOllama(model="llama3.2")
    prompt = hub.pull("langchain-ai/retrieval-qa-chat")
    docs = create_stuff_documents_chain(llm, prompt=prompt)
    retrieval_chain = create_retrieval_chain(retriever=retriever, combine_docs_chain=docs)

    return retrieval_chain


def scrape_webpages(file):
    
    with open(file, "r") as f:
        links = f.readlines()
        
        
        for url in links:
            url = url.strip('\n')
            page = requests.get(url).text

            soup = BeautifulSoup(page, features="html.parser")

            all_P = soup.find("div", class_="mw-body-content mw-content-ltr").find_all(["p", "li"])

            with open("scraped_pages/" + url.split('/')[-1] + ".txt", "w") as f:
                for text_block in all_P:
                    f.write(re.sub(r'\n+', '\n', text_block.text.strip().replace("	", "")))
            
            print(f"{url} preprocessed...")
        
    print("Preprocessing finished!")
    return

def load_files_to_pinecone():

    files = os.listdir("scraped_pages")
    for file in files:
        with open("scraped_pages/" + file, "r") as f:

            transcription = f.read()

            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
            docs = text_splitter.split_text(transcription)
            

            pc = Pinecone()

            if "miami" not in pc.list_indexes().names():
                pc.create_index("miami", dimension=768, spec=ServerlessSpec("aws", "us-east-1"))

            PineconeVectorStore.from_texts(texts=docs, embedding=embeddings, index_name="miami")
    return 


rag = create_chain()

with st.chat_message("ai"):
    prompt = st.chat_input("Ask any question about Hotline Miami Series:")
    if prompt:
        st.write(rag.invoke({"input" : prompt})["answer"])