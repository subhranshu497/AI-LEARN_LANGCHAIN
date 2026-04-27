from langchain_anthropic import AnthropicEmbeddings
from dotenv import load_dotenv
import os

#this code wont run as anthropic doesnt provide embeddings for the public yet. This is just a demo of how to use the AnthropicEmbeddings class once it is available.
print("Running Anthropic Model Embeddings Demo...")
# Load environment variables from .env file
load_dotenv()

# Initialize the AnthropicEmbeddings model
embedding_model = AnthropicEmbeddings(model="claude-haiku-4-5")

#document to be embedded

document = ["I love Chicago specially during spring and summer. The weather is perfect and the city is vibrant with activities."
"The food is amazing and there are so many things to do. I enjoy visiting the museums, parks, and the lakefront. The people are friendly and welcoming. "
"Overall, Chicago is a great city to live in or visit."]

# Generate embeddings for the document
embeddings = embedding_model.embed_documents(document)

print("Embeddings generated for the document:")
print(embeddings.content)