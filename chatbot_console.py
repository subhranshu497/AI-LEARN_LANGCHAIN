from langchain_anthropic import ChatAnthropic
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()
# Initialize the ChatAnthropic model with the API key from environment variables
model = ChatAnthropic(model="claude-haiku-4-5", max_tokens=100)

#build the bot
while True:
    user_input = input("Subh: ")
    if user_input.lower in ["exit", "quit"]:
        print ("LoriAI: Goodbye! Its been great chatting with you.")
        break
    response = model.invoke(user_input)
    print(f"LoriAI: {response.content}")
