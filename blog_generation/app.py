from urllib import response
import streamlit as st
from langchain.prompts import PromptTemplate
from langchain_community.llms import CTransformers




#get response from llama model

def getLLamaResponse(input_text, no_words, blog_style):
    llm = CTransformers(model = 'llama-2-7b-chat.ggmlv3.q8_0.bin',
                        model_type = 'llama',
                        config={'max_new_tokens':256,
                                'temperature':0.01})
    
    #prompt
    template = f""" Write a blog for {blog_style} job profile for a topic {input_text}
                    within {no_words} words."""
    prompt = PromptTemplate(input_variables=["blog_style","input_text","no_words"], template=template)
    
    #generate response from llama 2
    response = llm(prompt.format(blog_style=blog_style, input_text=input_text, n_words=no_words))    
    print(response)
    return response

st.set_page_config(page_title='Generate_Blogs',
                   layout='centered',
                   initial_sidebar_state='collapsed')
st.header('Generate Blogs')

input_text = st.text_input('Enter the Blog topic')

col1, col2 = st.columns([5, 5])

with col1:
    no_words = st.text_input('No of words')

with col2:
    blog_style = st.selectbox('Writing the blog for', ('Reseacrhers','Data Scientist', 'Common people'), index=0)

submit = st.button('Generate')

if submit:
    st.write(getLLamaResponse(input_text, no_words, blog_style))