from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from pathlib import Path
import config


def load_pdf(pdf_path: str) -> list[Document]:
    """
    Load PDF document and extract text with metadata

    Args:
        pdf_path: Path to the PDF file

    Returns:
        List of Document objects with page content and metadata
    """
    print(f"Loading PDF from: {pdf_path}")
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    print(f"Loaded {len(documents)} pages from PDF")
    return documents


def split_documents(documents: list[Document], chunk_size: int = None, chunk_overlap: int = None) -> list[Document]:
    """
    Split documents into smaller chunks for embedding

    Args:
        documents: List of Document objects to split
        chunk_size: Maximum size of each chunk (default from config)
        chunk_overlap: Overlap between chunks (default from config)

    Returns:
        List of split Document objects with metadata
    """
    chunk_size = chunk_size or config.CHUNK_SIZE
    chunk_overlap = chunk_overlap or config.CHUNK_OVERLAP

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""]
    )

    split_docs = text_splitter.split_documents(documents)

    for i, doc in enumerate(split_docs):
        if "chunk_index" not in doc.metadata:
            doc.metadata["chunk_index"] = i
        if "source" not in doc.metadata:
            doc.metadata["source"] = Path(documents[0].metadata.get("source", "unknown")).name

    return split_docs

def process_documents(manuals_dir: str = None) -> list[Document]:
    """
    Complete document processing pipeline for every manual in the manuals directory:
    load each PDF, tag its pages with which equipment it belongs to, then split.

    Args:
        manuals_dir: Directory containing the equipment manual PDFs (default from config)

    Returns:
        List of processed Document chunks (across all manuals) ready for embedding
    """

    manuals_dir = Path(manuals_dir or config.MANUALS_DIR)
    pdf_paths = sorted(manuals_dir.glob("*.pdf"))
    all_chunks = []

    for pdf_path in pdf_paths:
        print("pdf_path.stem", pdf_path.stem)
        documents = load_pdf(str(pdf_path))
        for doc in documents:
            doc.metadata["equipment"] = pdf_path.stem
        chunks = split_documents(documents)
        all_chunks.extend(chunks)
    return all_chunks

if __name__ == "__main__":
    chunks = process_documents()
    print(f"Total chunks: {len(chunks)}")