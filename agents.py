import time
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from langchain import hub
EMBEDDINGS = OllamaEmbeddings(model="nomic-embed-text")

def create_chain():
    vectorstore = PineconeVectorStore(index_name="miami", embedding=EMBEDDINGS)
    retriever = vectorstore.as_retriever()

    llm = ChatOllama(model="llama3.2")
    prompt = hub.pull("langchain-ai/retrieval-qa-chat")
    docs = create_stuff_documents_chain(llm, prompt=prompt)
    retrieval_chain = create_retrieval_chain(retriever=retriever, combine_docs_chain=docs)

    return retrieval_chain


def generate_output(response):
    
    for word in response.split():
        yield word + " "
        time.sleep(0.05)

