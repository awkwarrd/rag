import time
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

EMBEDDINGS = OllamaEmbeddings(model="nomic-embed-text")

SYSTEM_PROMPT = """
You are a helpful assistant. Communicate and answer any questions you've been asked based on your general knowledge, but if there is context provided below, use it to build your answer

Context: {context}

If there is no questions, just act friendly and try to communicate with a user.
"""

def create_prompt():
    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="messages")
     ])

    return prompt

def create_chain(prompt_func):
    vectorstore = PineconeVectorStore(index_name="miami", embedding=EMBEDDINGS)
    retriever = vectorstore.as_retriever(search_type="similarity_score_threshold", search_kwargs={"k": 3, "score_threshold": 0.8})

    llm = ChatOllama(model="llama3.2")
    prompt = prompt_func()
    docs = create_stuff_documents_chain(llm, prompt=prompt)
    retrieval_chain = create_retrieval_chain(retriever=retriever, combine_docs_chain=docs)

    return retrieval_chain


def generate_output(response):
    
    for word in response.split():
        yield word + " "
        time.sleep(0.05) 