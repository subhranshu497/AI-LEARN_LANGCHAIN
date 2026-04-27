from langchain_voyageai import VoyageAIEmbeddings
from dotenv import load_dotenv
from numpy import sort
from sklearn.metrics.pairwise import cosine_similarity
import os
print("Running Document Similarity Demo...")
# Load environment variables from .env file
load_dotenv()
# Initialize the OpenAIEmbeddings model
embedding_model = VoyageAIEmbeddings(
    voyage_api_key=os.getenv("VOYAGE_API_KEY"),
    model="voyage-4-lite",
    output_dimension=256                  
)

#documents to be embedded
document = [
    "Present-day international cricket is characterized by a high-octane balance between three distinct formats: "
    "Test matches, One Day Internationals, and T20s. The rapid rise of global franchise leagues has significantly influenced player "
    "availability and shifted the sport’s financial landscape toward the shorter versions of the game. Modern players now "
    "utilize advanced data analytics and innovative batting techniques to push the boundaries of traditional scoring rates. "
    "Additionally, the implementation of the Decision Review System (DRS) and enhanced broadcasting technology has brought a "
    "new level of precision to officiating and fan engagement. As the ICC expands the game into newer territories like the "
    "United States, the sport continues to evolve into a truly global entertainment phenomenon."
]
# Generate embeddings for the document
embeddings = embedding_model.embed_documents(document)
print("Embeddings generated for the document:")
print(f"Document Embeddings: {len(embeddings)}")
print(f"First 5 embeddings:{embeddings[:5]}")
# write a query to find its result similar to the document
query = "How has the rise of T20 cricket influenced the traditional formats of the game?"
# Generate embeddings for the query
query_embedding = embedding_model.embed_query(query)
print("Embedding generated for the query:")
print(f"Query Embedding: {len(query_embedding)}")
print(f"First 5 query embeddings: {query_embedding[:5]}")
# Calculate cosine similarity between the query embedding and document embeddings
similarity = cosine_similarity([query_embedding],embeddings)[0]
print("Cosine similarity between the query and document:")
index, score = sorted(list(enumerate(similarity)), key=lambda x: x[1])[-1]
print(query)
print("index ", index)
print("Similarity score: ", score)
