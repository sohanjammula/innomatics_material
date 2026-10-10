from langchain_ollama import OllamaEmbeddings

model = OllamaEmbeddings(model="nomic-embed-text")

sentence = "This is a test sentence for embedding generation."

embedding = model.embed_query(sentence)
print(f"Embedding for the sentence: {embedding}")
print(f"Length of the embedding: {len(embedding)}")