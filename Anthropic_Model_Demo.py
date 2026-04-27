from langchain_anthropic import ChatAnthropic

print("Running Anthropic Model Demo...")
from dotenv import load_dotenv
import os
# Load environment variables from .env file
load_dotenv()

# Initialize the AnthropicChat model
model = ChatAnthropic(model="claude-haiku-4-5", max_tokens=100)

prompt = "which team is leading the 2023 IPL season ?"

# Generate a response from the model
response = model.invoke(prompt)

print(response.content)

