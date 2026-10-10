from typing import List
from typing_extensions import TypedDict
from dotenv import load_dotenv

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import OpenAIEmbeddings
from langchain.chat_models import init_chat_model
from langchain_qdrant import QdrantVectorStore, RetrievalMode
from langgraph.graph import END, StateGraph, START
from qdrant_client import QdrantClient
from langchain_core.documents import Document
from langchain_community.utilities import SearchApiAPIWrapper
import config
from langchain_huggingface import HuggingFaceEmbeddings


load_dotenv()

search = SearchApiAPIWrapper()
class CRAG_State(TypedDict):
    """
    Represents the state of our CRAG graph.

    Attributes:
        question: question
        generation: LLM generation
        documents: list of documents
        times_transformed: number of times the question has been re-written
        should_transform_query: should we transform the query or not
    """

    question: str
    generation: str
    documents: List[str]
    times_transformed: int  # maintain the number of times the question has been re-written
    should_transform_query: str

def get_qdrant_client():
    """
    Create and return Qdrant Cloud client
    
    Returns:
        QdrantClient instance connected to Qdrant Cloud
    """
    client = QdrantClient(
        url=config.QDRANT_URL,
        api_key=config.QDRANT_API_KEY,
    )
    return client

def get_embeddings():
    """
    Initialize OpenAI embeddings model
    
    Returns:
        OpenAIEmbeddings instance
    """
    return HuggingFaceEmbeddings(model_name=config.EMBEDDING_MODEL)

def get_vector_store(collection_name: str = None):
    """
    Get or create Qdrant vector store
    
    Args:
        collection_name: Name of the collection (default from config)
        
    Returns:
        Qdrant vector store instance
    """
    collection_name = collection_name or config.QDRANT_COLLECTION_NAME
    client = get_qdrant_client()
    embeddings = get_embeddings()
    
    vector_store = QdrantVectorStore(
        client=client,
        collection_name=collection_name,
        embedding=embeddings,
        retrieval_mode=RetrievalMode.DENSE,
    )
    
    return vector_store


def get_retriever():
    """
    Get retriever from vector store
    
    Returns:
        Retriever instance
    """
    vector_store = get_vector_store()
    retriever = vector_store.as_retriever(search_kwargs={"k": config.RETRIEVAL_K})
    return retriever


def create_rag_chain():
    """
    Create RAG chain for answer generation
    
    Returns:
        RAG chain (prompt | llm | parser)
    """
    rag_prompt = '''
    You are a helpful customer support assistant for SolarNova Dynamics. 
    Your role is to answer questions about SolarNova Dynamics products, services, and business information based on the provided context from the company's business document.

    Guidelines:
    - Answer questions based ONLY on the provided context from the document
    - If the information is not in the context, politely say you don't have that information
    - Be friendly, professional, and helpful
    - Provide clear and concise answers
    - If asked about something not in the document, suggest contacting customer support directly

    Use the following context to answer the user's question:
    Question: {question} 
    Context: {context} 
    Answer:
    '''
    rag_prompt_template = ChatPromptTemplate.from_template(rag_prompt)
    rag_llm = init_chat_model(config.RAG_LLM_MODEL, model_provider=config.LLM_PROVIDER, temperature=config.RAG_LLM_TEMPERATURE)
    rag_chain = rag_prompt_template | rag_llm | StrOutputParser()
    return rag_chain

def create_question_rewriter():
    """
    Create question rewriter chain
    
    Returns:
        Question rewriter chain (prompt | llm | parser)
    """
    rewriter_llm = init_chat_model(config.REWRITER_LLM_MODEL, model_provider=config.LLM_PROVIDER, temperature=config.REWRITER_LLM_TEMPERATURE)
    rewriter_system = """You are a question re-writer that converts an input question to a better version that is optimized for RAG. Look at the input and try to reason about the underlying semantic intent / meaning."""
    re_write_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", rewriter_system),
            (
                "human",
                "Here is the initial question: \n\n {question} \n Formulate an improved question.",
            ),
        ]
    )
    question_rewriter = re_write_prompt | rewriter_llm | StrOutputParser()
    return question_rewriter


def create_retrieval_grader():
    """
    Create retrieval grader chain
    
    Returns:
        Retrieval grader chain (prompt | llm | boolean parser)
    """
    grader_llm = init_chat_model(config.GRADER_LLM_MODEL, model_provider=config.GRADER_LLM_PROVIDER)
    grader_system = """Assess the relevance of the retrieved document to the user's question. Respond only with 'yes' if the document provides information that directly answers or helps answer the question; otherwise, respond with 'no'."""
    grade_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", grader_system),
            ("human", "Retrieved document: \n\n {document} \n\n User question: {question}"),
        ]
    )
    retrieval_grader = grade_prompt | grader_llm | StrOutputParser()
    return retrieval_grader

#initalize the chains
rag_chain = create_rag_chain()
question_rewriter = create_question_rewriter()
retrieval_grader = create_retrieval_grader()
retriever = get_retriever()

#Graph
#Retrieval Node
def retrieve(state):
    """
    Retrieve documents

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): New key added to state, documents, that contains retrieved documents
    """
    print("Retrieving documents")
    question = state["question"]

    # Retrieval
    documents = retriever.invoke(question)
    return {"documents": documents}


def grade_documents(state):
    """
    Determines whether the retrieved documents are relevant to the question.

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): Updates documents key with only filtered relevant documents
    """
    print("Check Document Relevance To Question")
    question = state["question"]
    documents = state["documents"]
    times_transformed = state["times_transformed"]

    # Score each doc
    filtered_docs = []
    should_transform_query = "No"
    for d in documents:
        score = retrieval_grader.invoke(
            {"question": question, "document": d.page_content}
        )
        if score == "yes":
            
            filtered_docs.append(d)
    if len(filtered_docs) == 0:
        print("Grade: Documents Not Relevant")
        should_transform_query = "Yes"
    else:
        print(f"Found {len(filtered_docs)} relevant documents proceed to generate answer")

    if len(filtered_docs) == 0 and times_transformed >= 3:
       
        should_transform_query = "No"

        search_results = search.run(question)
        print("Search Results:")
        print(search_results)
        filtered_docs = [Document(page_content=search_results)]

    return {"documents": filtered_docs, "should_transform_query": should_transform_query}


def transform_query(state):
    """
    Transform the query to produce a better question.

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): Updates question key with a re-phrased question
    """
    print("Transform Query")
    question = state["question"]
    times_transformed = state["times_transformed"]
    
        
    times_transformed += 1

    # Re-write question
    better_question = question_rewriter.invoke({"question": question})
    print('New Question:')
    print(better_question)
    return {"question": better_question, "times_transformed": times_transformed}



def format_docs(docs):
    """Format documents for context"""
    return "\n\n".join(doc.page_content for doc in docs)

def generate(state):
    """
    Generate answer

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): New key added to state, generation, that contains LLM generation
    """
    print("Generate")
    question = state["question"]
    documents = state["documents"]

    # RAG generation
    generation = rag_chain.invoke({"context": format_docs(documents), "question": question})
    return {"generation": generation}




#Build the Graph

crag_workflow = StateGraph(CRAG_State)

#Add the nodes to the graph


# Add nodes
crag_workflow.add_node("retrieve", retrieve)
crag_workflow.add_node("grade_documents", grade_documents)
crag_workflow.add_node("generate", generate) 
crag_workflow.add_node("transform_query", transform_query)


#Add the edges-- connections between the nodes


def decide_gen(state):
    if state["should_transform_query"] == "No":
        return "generate"
    else:
        return "transform"

crag_workflow.add_edge(START, "retrieve")
crag_workflow.add_edge("retrieve", "grade_documents")

#lambda x :"generate" if x["should_transform_query"]=="No" else "transform"
crag_workflow.add_conditional_edges("grade_documents", decide_gen, 
                                    {"generate":"generate","transform":"transform_query"})
crag_workflow.add_edge("transform_query", "retrieve")
crag_workflow.add_edge("generate", END)

crag_workflow = crag_workflow.compile()

crag_workflow.get_graph().draw_mermaid_png(output_file_path="CRAG_Langgraph.png")


question = "What is AI?"
inputs = {"question": question, "times_transformed": 0}
final_state = None
    
for output in crag_workflow.stream(inputs):
    for key, value in output.items():
        # Node
        print(f"Node '{key}': {value}")
        final_state = value

print("Final State:")
print(final_state)