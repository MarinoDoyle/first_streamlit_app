import streamlit as st 
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import GPT2Tokenizer, GPT2Model, pipeline
from langchain.llms import HuggingFacePipeline
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain_core.runnables import RunnablePassthrough
from langchain.schema import StrOutputParser
from langchain import hub

tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2Model.from_pretrained('gpt2')
text = "Replace me by any text you'd like."
encoded_input = tokenizer(text, return_tensors='pt')
output = model(**encoded_input)

# generator = pipeline('text-generation', model='gpt2')
# st.write(generator("What is Machine Learning", num_return_sequences=5))

# Loading in the CSV files: 

loader = CSVLoader(file_path='work_dummy_data.csv')
data = loader.load()

text_splitter_csv = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=20)
all_splits_csv = text_splitter_csv.split_documents(data)

# st.write(all_splits_csv)

embeddings = HuggingFaceEmbeddings()
vectorstore = FAISS.from_documents(all_splits_csv, embeddings)
question = "What is the most frequent result for total number of rooms"
embedding_vector = embeddings.embed_query(question)
st.write(embedding_vector)

retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 6})

prompt = hub.pull("rlm/rag-prompt")

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | model
    | StrOutputParser()
)

rag_chain.invoke("What are some key insights?")
