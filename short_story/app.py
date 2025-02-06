import streamlit as st
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

model_name = 'meta-llama/Llama-2-7b-hf'
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

st.title('Short Story Writer')
prompt = st.text_area("Your Story Prompt", "Once upon a time in a distant land...")
max_length = st.slider("Story Length (max tokens)", min_value=50, max_value=1000, value=300)

if st.button("Generate"):
    with st.spinner('Writing your story'):
        inputs = tokenizer(prompt, return_tensors="pt")
        outputs = model.generate(
            **inputs,
            max_length=max_length,
            num_return_sequences=1,
            do_sample=True,
            temperature=0.8,
            top_p=0.95
        )
        story = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
    st.subheader('Generate story')
    st.write(story)
        
