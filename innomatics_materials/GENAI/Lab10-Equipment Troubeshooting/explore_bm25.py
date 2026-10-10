from fastembed import SparseTextEmbedding


# Initialize BM25 model
bm25 = SparseTextEmbedding(model_name="Qdrant/bm25")

documents = [
    "The quick brown fox jumps over the lazy dog Machine",
    "Machine learning is transforming artificial intelligence",
    "Python is a popular programming language",
    "The fox is clever and quick"
]

doc_embeddings = list(bm25.embed(documents))

for i, emb in enumerate(doc_embeddings):
    print(f"Document {i}")
    print(emb.as_dict())
    print("Indices:", emb.indices[:10])
    print("Values :", emb.values[:10])

# query_res = list(bm25.query_embed(documents))


# print("Query : ",query_res)