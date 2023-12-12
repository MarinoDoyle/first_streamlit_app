import streamlit as st 
from transformers import AutoTokenizer, AutoModelForCausalLM
import transformers
from langchain.llms import HuggingFacePipeline

tokenizer = AutoTokenizer.from_pretrained("llmware/bling-falcon-1b-0.1")
model = AutoModelForCausalLM.from_pretrained("llmware/bling-falcon-1b-0.1")


generate_text = transformers.pipeline(
    model=model,
    tokenizer=tokenizer,
    return_full_text=True,  # langchain expects the full text
    task='text-generation',
    # temperature=0.1,  # 'randomness' of outputs, 0.0 is the min and 1.0 the max
    max_new_tokens=512,  # max number of tokens to generate in the output
    repetition_penalty=1.1  # without this output begins repeating
)

res = generate_text("Explain me the difference a Server and a Client")
st.write((res[0]["generated_text"]))

