from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, PromptTemplate
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_community.chat_message_histories import RedisChatMessageHistory

from langchain.chat_models import init_chat_model
from langchain_core.runnables import RunnablePassthrough
from langchain_qdrant import QdrantVectorStore
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
import config
from vector_store_setup import get_vector_store

REDIS_URL = config.REDIS_URL

REDIS_KEY_PREFIX = config.REDIS_KEY_PREFIX

def get_redis_history(session_id: str) :
    return RedisChatMessageHistory(
        session_id=session_id,
        url=REDIS_URL,
        key_prefix=REDIS_KEY_PREFIX,
    )



def create_rag_chain(vector_store: QdrantVectorStore = None):
    """
    Create RAG chain using LangChain's built-in create_stuff_documents_chain.

    Retrieval runs in hybrid mode (dense + sparse fused by Qdrant), and each
    retrieved chunk is prefixed with its source manual + page so the model
    (and the user) can tell which manual an answer came from.

    Args:
        vector_store: Qdrant vector store instance (if None, will load from config)

    Returns:
        RAG chain ready to be used with message history
    """
    if vector_store is None:
        vector_store = get_vector_store()

    retriever = vector_store.as_retriever(
        search_kwargs={"k": config.RETRIEVAL_K}
    )

   #ollama
    if not config.LLM_BASE_URL:
        llm = init_chat_model(config.LLM_MODEL, model_provider=config.LLM_PROVIDER)
    else:
        llm = init_chat_model(config.LLM_MODEL, model_provider=config.LLM_PROVIDER, base_url=config.LLM_BASE_URL, api_key = config.LLM_API_KEY)
   
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are an equipment troubleshooting assistant for warehouse conveyor systems.
You have access to the installation, operation, maintenance and parts manuals for these units:
TGW IntelliROL Motorized Roller Conveyor, 2200 Series Conveyors, COESIA FlexLink X85X, and
Interroll BM 8350.

Guidelines:
- Answer questions based ONLY on the provided manual excerpts below
- Each excerpt is labeled with the equipment and page it came from - mention which
  equipment/page your answer is based on
- If the manuals don't cover something, say so plainly instead of guessing
- Be clear and concise, and use step-by-step instructions for troubleshooting/maintenance steps

Manual excerpts:
{context}"""),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}")
    ])

    # Prefixes each stuffed chunk with its equipment + page, e.g.
    # "[TGW IntelliROL Motorized Roller Conveyer, page 6]\n<chunk text>"
    document_prompt = PromptTemplate.from_template(
        "[{equipment}, page {page}]\n{page_content}"
    )

    combine_docs_chain = create_stuff_documents_chain(
        llm=llm,
        prompt=prompt,
        document_variable_name="context",
        document_prompt=document_prompt,
    )

    def get_context(x):
        # print("Relevant Chunks:")
        context = retriever.invoke(x.get("input", "")) #Retrieve relevant chunks
        # for chunk in context:
        #     print("Chunk Content: ", chunk.page_content)
        #     print("--------------------------------")
        #     print("Chunk Metadata: ", chunk.metadata)
        #     print("--------------------------------")
        # print("--------------------------------")
        return context

    rag_chain = (
        RunnablePassthrough.assign(
            context=get_context
        )
        | combine_docs_chain
    )

    chain_with_history = RunnableWithMessageHistory(
        rag_chain,
        get_redis_history,
        input_messages_key="input",
        history_messages_key="chat_history"
    )
    return chain_with_history

