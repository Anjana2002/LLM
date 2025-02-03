import streamlit as st
from transformers import T5ForConditionalGeneration, T5Tokenizer

# Load the T5-small model and tokenizer
@st.cache_resource
def load_model():
    model_name = "t5-small"
    model = T5ForConditionalGeneration.from_pretrained(model_name)
    tokenizer = T5Tokenizer.from_pretrained(model_name)
    return model, tokenizer

model, tokenizer = load_model()

# Streamlit app
st.title("Text Summarization with T5-Small")

# Input text
input_text = st.text_area("Enter your text here:")

# Summarize button
if st.button("Summarize"):
    if input_text:
        # Tokenize and generate summary
        inputs = tokenizer.encode("summarize: " + input_text, return_tensors="pt", max_length=512, truncation=True)
        summary_ids = model.generate(inputs, max_length=50, min_length=25, length_penalty=2.0, num_beams=4, early_stopping=True)
        summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

        # Display the summary
        st.write("Summary:")
        st.write(summary)
    else:
        st.write("Please enter some text to summarize.")