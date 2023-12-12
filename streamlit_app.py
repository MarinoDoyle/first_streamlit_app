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
import tempfile
from langchain.chains import ConversationalRetrievalChain

api_key = st.sidebar.text_input(
    label="#### Your OpenAI API key 👇",
    placeholder="Paste your openAI API key, sk-",
    type="password")

# uploaded_file = st.sidebar.file_uploader("upload", type="csv")
file_name = "work_dummy_data"

qa_results = []

if api_key:
    # Creating a csv agent allows us to query the tables a lot easier
    agent = create_csv_agent(
        ChatOpenAI(temperature=0,  api_key=api_key),
        "work_dummy_data.csv",
        verbose=True,
        agent_type=AgentType.OPENAI_FUNCTIONS,
    )

    # Loading in the actual data.
    loader = CSVLoader(file_path="work_dummy_data.csv", encoding="utf-8", csv_args={
                'delimiter': ','})
    data = loader.load()
    st.write(data)

    embeddings = OpenAIEmbeddings(api_key=api_key)
    vectorstore = FAISS.from_documents(data, embeddings)

    chain = ConversationalRetrievalChain.from_llm(
    llm = ChatOpenAI(temperature=0.0,model_name='gpt-3.5-turbo', api_key=api_key),
    retriever=vectorstore.as_retriever())


def conversational_chat(query):
        
    result = chain({"question": query, 
    "chat_history": st.session_state['history']})
    st.session_state['history'].append((query, result["answer"]))    
    return result["answer"]



st.title("Chat-Based Language Model")

question = st.text_input("Enter your question here:")

if st.button("Ask"):
    if 'api_key' not in locals():
        st.error("Please enter your API Key in the sidebar.")
    else:
        st.subheader("Stored Q&A Results:")
        for idx, (q, a) in enumerate(qa_results, 1):
            st.write(f"Q{idx}: {q}")
            st.write(f"A{idx}: {a}")
            # response = chain.run(question)
            # st.text_area("Response:", value=response)
