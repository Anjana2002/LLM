import streamlit as st
from langchain.prompts import PromptTemplate
from langchain_community.llms import CTransformers

# Function to summarize content
def summarize_content(input_text, no_words, blog_style):
    llm = CTransformers(model='/home/anjana/Project/LLM/llama-2-7b-chat.ggmlv3.q8_0.bin', 
                        model_type='llama',
                        config={'max_new_tokens': 150, 'temperature': 0.01})
    
    # Prepare the prompt
    template = f"""Write a summary of the content about {input_text} for {blog_style} job profile, within {no_words} words."""
    prompt = PromptTemplate(input_variables=["blog_style", "input_text", "no_words"], template=template)
    
    # Generate response from LLaMA
    response = llm(prompt.format(blog_style=blog_style, input_text=input_text, no_words=no_words))
    return response

# Streamlit UI setup
st.set_page_config(page_title='Summarize Content', layout='centered', initial_sidebar_state='collapsed')
st.header('Summarize Content')

# Input fields for user to enter content
input_text = st.text_area('Enter the content to summarize', height=150)

col1, col2 = st.columns([5, 5])
with col1:
    no_words = st.text_input('Number of words for summary')

with col2:
    blog_style = st.selectbox('Writing style for summary', ('Researchers', 'Data Scientist', 'Common people'), index=0)

# Button to trigger summary generation
submit = st.button('Generate Summary')

if submit:
    # Check if inputs are valid
    if not input_text:
        st.warning("Please enter some content to summarize.")
    elif not no_words.isdigit() or int(no_words) <= 0:
        st.warning("Please enter a valid number for the number of words.")
    else:
        st.write("Generating summary...")
        with st.spinner("Please wait..."):
            # Generate the summary
            result = summarize_content(input_text, no_words, blog_style)
        st.text_area('Generated Summary:', result, height=300)
