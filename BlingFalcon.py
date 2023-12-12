import streamlit as st 
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import GPT2Tokenizer, GPT2Model
from langchain.llms import HuggingFacePipeline

tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2Model.from_pretrained('gpt2')
text = "Replace me by any text you'd like."
encoded_input = tokenizer(text, return_tensors='pt')
st.write(output = model(**encoded_input))


# generate_text = transformers.pipeline(
#     model=model,
#     tokenizer=tokenizer,
#     return_full_text=True,  # langchain expects the full text
#     task='text-generation',
#     # temperature=0.1,  # 'randomness' of outputs, 0.0 is the min and 1.0 the max
#     max_new_tokens=512,  # max number of tokens to generate in the output
#     repetition_penalty=1.1  # without this output begins repeating
# )

# res = generate_text("Explain me the difference a Server and a Client")
# st.write((res[0]["generated_text"]))

