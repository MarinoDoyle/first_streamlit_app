import streamlit as st 
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import GPT2Tokenizer, GPT2Model, pipeline
from langchain.llms import HuggingFacePipeline
from langchain.document_loaders.csv_loader import CSVLoader

tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2Model.from_pretrained('gpt2')
text = "Replace me by any text you'd like."
encoded_input = tokenizer(text, return_tensors='pt')
output = model(**encoded_input)

generator = pipeline('text-generation', model='gpt2')
st.write(generator("What is Machine Learning", num_return_sequences=5))

# Loading in the CSV files: 

loader = CSVLoader(file_path='work_dummy_data.csv')
data = loader.load()

text_splitter_csv = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=20)
all_splits_csv = text_splitter_csv.split_documents(data)

st.write(all_splits_csv)
