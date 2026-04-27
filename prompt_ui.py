from langchain_anthropic import ChatAnthropic
import streamlit as st
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()
# Initialize the ChatAnthropic model with the API key from environment variables

model = ChatAnthropic(model="claude-haiku-4-5", max_tokens=100)

#build the ui using streamlit
st.header("Anthropic Model Demo")
prompt = st.text_input("Enter your prompt here:")

if st.button("Generate Response"):
    result = model.invoke(prompt)
    st.write(result.content)


