import os
import boto3
import streamlit as st

# Set AWS credentials
os.environ['AWS_ACCESS_KEY_ID'] = ''
os.environ['AWS_SECRET_ACCESS_KEY'] = ''
os.environ['AWS_REGION'] = 'us-east-1' 

# Updated Imports
from langchain_community.embeddings import BedrockEmbeddings
from langchain_community.llms import Bedrock
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_community.vectorstores import FAISS

# Data Ingestion
import numpy as np
from langchain.text_splitter import RecursiveCharacterTextSplitter

# LLM Models
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA

# Bedrock Clients
bedrock = boto3.client(service_name="bedrock-runtime", region_name="us-east-1")
bedrock_embeddings = BedrockEmbeddings(model_id="amazon.titan-embed-text-v1", client=bedrock)

# Data ingestion
def data_ingestion():
    loader = PyPDFDirectoryLoader("data")
    documents = loader.load()
    print(f"Loaded {len(documents)} documents.")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    docs = text_splitter.split_documents(documents)
    print(f"Split into {len(docs)} chunks.")
    return docs

# Vector Embedding and vector store
def get_vector_store(docs):
    print("Generating embeddings for document chunks...")
    texts = [doc.page_content for doc in docs]  # Extract texts from documents
    embeddings = bedrock_embeddings.embed_documents(texts)  # Generate embeddings
    if len(embeddings) != len(docs):
        raise ValueError(f"Number of embeddings ({len(embeddings)}) does not match number of documents ({len(docs)}).")
    print(f"Generated {len(embeddings)} embeddings.")

    vectorstore_faiss = FAISS.from_texts(texts, bedrock_embeddings)
    vectorstore_faiss.save_local("faiss_index")

def get_llama2_llm():
    llm = Bedrock(model_id="meta.llama3-70b-instruct-v1:0", client=bedrock, model_kwargs={'max_gen_len': 512})
    return llm

prompt_template = """
Human: Use the following pieces of context to provide a 
concise answer to the question at the end but use at least summarize with 
250 words with detailed explanations. If you don't know the answer, 
just say that you don't know, don't try to make up an answer.
<context>
{context}
</context>

Question: {question}

Assistant:"""

PROMPT = PromptTemplate(
    template=prompt_template, input_variables=["context", "question"]
)

def get_response_llm(llm, vectorstore_faiss, query):
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore_faiss.as_retriever(search_type="similarity", search_kwargs={"k": 3}),
        return_source_documents=True,
        chain_type_kwargs={"prompt": PROMPT}
    )
    answer = qa({"query": query})
    return answer['result']

def main():
    st.set_page_config("Chat PDF")

    st.header("CC Project : Chat with PDF using AWS Bedrock and IAM")

    user_question = st.text_input("Ask a Question from the PDF Files")

    with st.sidebar:
        st.title("Update Or Create Vector Store:")

        if st.button("Vectors Update"):
            with st.spinner("Processing..."):
                docs = data_ingestion()
                get_vector_store(docs)
                st.success("Done")


    if st.button("Output"):
        with st.spinner("Processing..."):
            faiss_index = FAISS.load_local("faiss_index", bedrock_embeddings, allow_dangerous_deserialization=True)
            llm = get_llama2_llm()
            st.write(get_response_llm(llm, faiss_index, user_question))
            st.success("Done")

if __name__ == "__main__":
    main()
