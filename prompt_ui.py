from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate
import streamlit as st
from dotenv import load_dotenv
import os
import json

# Load environment variables from .env file
load_dotenv()
# Initialize the ChatAnthropic model with the API key from environment variables

model = ChatAnthropic(model="claude-haiku-4-5", max_tokens=10000)

#build the ui using streamlit
st.header("Research Assistant")
paper_input = st.selectbox( 
    "Select Anthropic Research Paper", 
    [
        "A Mathematical Framework for Transformer Circuits", 
        "Constitutional AI: Harmlessness from AI Feedback", 
        "Scaling Monosemanticity: Extracting Interpretable Features", 
        "The Claude 3 Model Family: Opus, Sonnet, Haiku",
        "Circuit Tracing: Revealing Computational Graphs in LLMs"
    ] 
)

style_input = st.selectbox( "Select Explanation Style", ["Beginner-Friendly", "Technical", "Code-Oriented", "Mathematical"] ) 

length_input = st.selectbox( "Select Explanation Length", ["Short (1-2 paragraphs)", "Medium (3-5 paragraphs)", "Long (detailed explanation)"] )

with open("staticResources/template.json") as f:
    template_str = json.load(f)["template"]
template = ChatPromptTemplate.from_template(template_str)

if st.button("Summarize"):
    chain = template | model
    prompt = f"Paper: {paper_input}\nStyle: {style_input}\nLength: {length_input}"
    result = chain.invoke(prompt)
    st.write(result.content)
    


