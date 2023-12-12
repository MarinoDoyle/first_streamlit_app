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


api_key = st.sidebar.text_input(
    label="#### Your OpenAI API key 👇",
    placeholder="Paste your openAI API key, sk-",
    type="password")

# uploaded_file = st.sidebar.file_uploader("upload", type="csv")

if api_key:
    agent = create_csv_agent(
        ChatOpenAI(temperature=0,  api_key=api_key),
        "work_dummy_data.csv",
        verbose=True,
        agent_type=AgentType.OPENAI_FUNCTIONS,
    )


    loader = CSVLoader(file_path="survey.csv", encoding="utf-8", csv_args={
                'delimiter': ','})
    data = loader.load()

    st.write(data)


st.title("Chat-Based Language Model")

question = st.text_input("Enter your question here:")

if st.button("Ask"):
    if 'api_key' not in locals():
        st.error("Please enter your API Key in the sidebar.")
    else:
        response = agent.run(question)
        st.text_area("Response:", value=response)
