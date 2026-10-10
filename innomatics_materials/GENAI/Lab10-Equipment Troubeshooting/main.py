import sys
from pathlib import Path
import config
from document_processor import process_documents
from vector_store_setup import store_documents, collection_exists, get_qdrant_client, create_collection_if_not_exists
from chatbot import create_rag_chain
import uuid


def setup_vector_store():
    """
    Setup vector store: check if it exists, if not process and store documents
    """
    print("=" * 60)
    print("Equipment Troubleshooting RAG - Vector Store Setup")
    print("=" * 60)

    manuals_dir = Path(config.MANUALS_DIR)
    if not manuals_dir.exists():
        print(f"Error: Manuals directory not found at {manuals_dir}")
        print("Please ensure the equipment manual PDFs are in that directory.")
        sys.exit(1)

    client = get_qdrant_client()
    collection_exists_flag = collection_exists(client, config.QDRANT_COLLECTION_NAME)

    if collection_exists_flag:
        collection_info = client.get_collection(config.QDRANT_COLLECTION_NAME)
        if collection_info.points_count > 0:
            print(f"Vector store already exists with {collection_info.points_count} documents")
            print("Skipping document processing...")
            return
    else:
        create_collection_if_not_exists(client, config.QDRANT_COLLECTION_NAME)

    print("Processing equipment manuals...")
    chunks = process_documents()

    print("\nStoring documents in Qdrant Cloud...")
    store_documents(chunks)
    print("Vector store setup complete!\n")


def main():
    print("=" * 60)
    print("Equipment Troubleshooting Assistant")
    print("=" * 60)
    print()
    session_id = str(uuid.uuid4())

    try:
        setup_vector_store()
    except Exception as e:
        print(f"Error setting up vector store: {e}")
        print("Please check your Qdrant Cloud credentials in .env file")
        sys.exit(1)

    print("Initializing chatbot...")
    try:
        chatbot = create_rag_chain()
        chat_config = {"configurable": {"session_id": session_id}}
        print("Chatbot ready!\n")
    except Exception as e:
        print(f"Error initializing chatbot: {e}")
        sys.exit(1)

    print("=" * 60)
    print("Chat Interface")
    print("Ask troubleshooting/maintenance questions about the conveyor equipment")
    print("Commands: 'quit' or 'exit' to end")
    print("=" * 60)
    print()

    while True:
        try:
            user_input = input("You: ").strip()

            if user_input.lower() in ['quit', 'exit', 'q']:
                print("\nGoodbye!")
                break

            if not user_input:
                continue

            print("Bot: ", end="", flush=True)
            response = chatbot.invoke(
                {"input": user_input},
                config=chat_config
            )
            print(response)
            print()

        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break
        except Exception as e:
            print(f"\nError: {e}")
            print("Please try again or type 'quit' to exit.\n")


if __name__ == "__main__":
    main()
