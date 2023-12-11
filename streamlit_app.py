import streamlit
import pandas 
import requests
import snowflake.connector
from urllib.error import URLError
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough
from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain.document_loaders.csv_loader import CSVLoader

# Using an OPENAI key
os.environ['OPENAI_API_KEY'] = "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"

llm = ChatOpenAI()

streamlit.title("Chat-Based Language Model")

question = streamlit.text_input("Enter your question here:")

if streamlit.button("Ask"):
    loader = CSVLoader(file_path=r"survey.csv")
    data = loader.load()

    text_splitter_csv = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=20)
    all_splits_csv = text_splitter_csv.split_documents(data)

    vector_store = FAISS.from_documents(all_splits_csv, OpenAIEmbeddings())

    retriever = vector_store.as_retriever()

    template = """Answer the question based only on the following context:
    {context}
    
    Question: {question}
    """
    prompt = ChatPromptTemplate.from_template(template)

    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    response = chain.invoke(question)
    streamlit.text_area("Response:", value=response)


