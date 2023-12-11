import streamlit as st
import os
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import LLMChain
from langchain.llms import OpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough
from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI

st.sidebar.title("Enter API Key")
api_key = st.sidebar.text_input("API Key:")

if api_key:
    os.environ['OPENAI_API_KEY'] = api_key

st.title("Chat-Based Language Model")

question = st.text_input("Enter your question here:")

if st.button("Ask"):
    if 'OPENAI_API_KEY' not in os.environ:
        st.error("Please enter your API Key in the sidebar.")
    else:
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

        llm = ChatOpenAI()

        chain = (
            {"context": retriever, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )

        response = chain.invoke(question)
        st.text_area("Response:", value=response)
