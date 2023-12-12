import streamlit as st 
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import GPT2Tokenizer, GPT2Model, pipeline
from langchain.llms import HuggingFacePipeline

tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2Model.from_pretrained('gpt2')
text = "Replace me by any text you'd like."
encoded_input = tokenizer(text, return_tensors='pt')
output = model(**encoded_input)
#st.write(output)
# llm = 
generator = pipeline('text-generation', model='gpt2')
st.write(generator("What is Machine Learning", max_length=10, num_return_sequences=5))
