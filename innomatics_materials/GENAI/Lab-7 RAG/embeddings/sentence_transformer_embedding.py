# from sentence_transformers import SentenceTransformer

from langchain_huggingface import HuggingFaceEmbeddings

from sklearn.metrics.pairwise import cosine_similarity
from visualization_utils import visualize_embeddings_3d
import numpy as np


model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
# sentence = "This is a test sentence for embedding generation."


# embedding = model.encode(sentence)
# print(f"Embedding for the sentence: {embedding}")
# print(f"Length of the embedding: {len(embedding)}")


# Example texts
texts = [
    "Machine learning algorithms",
    "Deep learning neural networks", 
    "Natural language processing",
    "Computer vision applications",
    "Data science techniques",
    "AI is the future",
    "I love coffee",
    "Machine learning algorithms"
]

tech_embeddings = model.embed_documents(texts) #384
embeddings_array = np.array(tech_embeddings)

print(f"Embeddings shape: {embeddings_array.shape}")  # Should print (8, 1536)

# Calculate similarities using cosine similarity
similarities = cosine_similarity(embeddings_array)

print("\nSimilarity Matrix:")
print(similarities)

print("\nSentence Similarities:")
for i, text1 in enumerate(texts):
    for j, text2 in enumerate(texts):
        if i < j:
            print(f"'{text1}' vs '{text2}': {similarities[i][j]:.3f}")

# Visualize embeddings in 3D
visualize_embeddings_3d(tech_embeddings, texts, title="Sentence Transformer Embeddings - 3D Visualization")
