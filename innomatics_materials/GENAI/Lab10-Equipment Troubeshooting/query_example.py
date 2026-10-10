from langchain_qdrant import QdrantVectorStore, RetrievalMode
import config
from vector_store_setup import get_qdrant_client, get_dense_embeddings, get_sparse_embeddings



query = "What is process to fix the belt issue?"

client = get_qdrant_client()
dense_embeddings = get_dense_embeddings()
sparse_embeddings = get_sparse_embeddings()

dense_store = QdrantVectorStore(
        client=client,
        collection_name=config.QDRANT_COLLECTION_NAME,
        embedding=dense_embeddings,
        retrieval_mode=RetrievalMode.DENSE,
        vector_name=config.DENSE_VECTOR_NAME,
    )

sparse_store = QdrantVectorStore(
        client=client,
        collection_name=config.QDRANT_COLLECTION_NAME,
        sparse_embedding=sparse_embeddings,
        retrieval_mode=RetrievalMode.SPARSE,
        sparse_vector_name=config.SPARSE_VECTOR_NAME,
    )


hybrid_store = QdrantVectorStore(
        client=client,
        collection_name=config.QDRANT_COLLECTION_NAME,
        embedding=dense_embeddings,
        sparse_embedding=sparse_embeddings,
        retrieval_mode=RetrievalMode.HYBRID,
        vector_name=config.DENSE_VECTOR_NAME,
        sparse_vector_name=config.SPARSE_VECTOR_NAME,
    )

def print_results(title, docs):
    print(f"\n--- {title} ---")
    for doc in docs:
        print(f"[{doc.metadata.get('equipment')}, page {doc.metadata.get('page')}]")
        print(doc.page_content[:200].replace("\n", " "))
        print()

print_results("Dense (semantic) search", dense_store.similarity_search(query, k=config.RETRIEVAL_K))
print_results("Sparse (BM25 keyword) search", sparse_store.similarity_search(query, k=config.RETRIEVAL_K))

print_results("Hybrid search", hybrid_store.similarity_search(query, k=config.RETRIEVAL_K))