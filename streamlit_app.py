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
from langchain_experimental.agents.agent_toolkits import create_csv_agent
from langchain.agents.agent_types import AgentType


st.sidebar.title("Enter API Key")
api_key = st.sidebar.text_input("API Key:")

if api_key:
    agent = create_csv_agent(
        ChatOpenAI(temperature=0,  api_key=api_key),
        "work_dummy_data.csv",
        verbose=True,
        agent_type=AgentType.OPENAI_FUNCTIONS,
    )
# Creating text embeddings and vector database
loader = CSVLoader(file_path="work_dummy_data.csv")
data = loader.load()

text_splitter_csv = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=20)
all_splits_csv = text_splitter_csv.split_documents(data)

embeddings = OpenAIEmbeddings(api_key=api_key)
vector_store = FAISS.from_documents(all_splits_csv, embeddings)

st.title("Chat-Based Language Model")

question = st.text_input("Enter your question here:")

if st.button("Ask"):
    if 'csv_agent' not in locals():
        st.error("Please enter your API Key in the sidebar.")
    else:
        response = csv_agent.ask(question)
        st.text_area("Response from CSV agent:", value=response)

        # Utilizing embeddings and vector database
        if 'vector_store' in locals():
            query_embedding = embeddings.encode_text(question)
            retrieved = vector_store.retrieve(query_embedding)
            st.text("Retrieved similar documents:", retrieved)
