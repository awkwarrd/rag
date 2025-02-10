import os
import re
from bs4 import BeautifulSoup
from langchain_ollama import OllamaEmbeddings
from langchain_pinecone import Pinecone, PineconeVectorStore
from pinecone import ServerlessSpec
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import YoutubeLoader
import requests



EMBEDDINGS = OllamaEmbeddings(model="nomic-embed-text")


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

    PineconeVectorStore.from_documents(documents=docs, embedding=EMBEDDINGS, index_name="miami")
    return 

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

            PineconeVectorStore.from_texts(texts=docs, embedding=EMBEDDINGS, index_name="miami")
    return 

