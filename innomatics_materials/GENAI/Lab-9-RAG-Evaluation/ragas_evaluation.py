import json
import sys
from pathlib import Path

# Add Lab-8 directory to path to import modules
lab8_path = Path(__file__).parent.parent / "Lab-8-Customer Service Chatbot"
sys.path.insert(0, str(lab8_path))

import config
from vector_store_setup import get_vector_store, get_embeddings
from customer_support_chat import create_rag_chain
import uuid

import pandas as pd
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
)
from ragas.dataset_schema import EvaluationDataset
from ragas.llms import llm_factory
from typing import List, Dict
from tqdm import tqdm #progress bar
from openai import OpenAI
from ragas.embeddings import LangchainEmbeddingsWrapper


import os
from dotenv import load_dotenv

load_dotenv()

# Setup OpenAI client
client = OpenAI(api_key = os.getenv("GROQ_API_KEY"), base_url = "https://api.groq.com/openai/v1" )

# Setup LLM for evaluation
# max_tokens is bumped from the ragas default (1024) because the legacy metrics
# (faithfulness, context_precision, context_recall) ask the LLM for verbose
# structured JSON output that gets truncated at the default limit.
llm = llm_factory("openai/gpt-oss-120b", client=client, max_tokens=4096)

# Setup embeddings for AnswerRelevancy metric
# The legacy ResponseRelevancy metric calls embeddings.embed_query/embed_documents
# (the langchain Embeddings interface), so the langchain-based embeddings used for
# the vector store are wrapped rather than using ragas' new-style HuggingFaceEmbeddings
# (which only exposes embed_text/embed_texts).
embeddings = LangchainEmbeddingsWrapper(get_embeddings())


#Step -1 Load the test cases

def load_test_cases(json_file_path: str) -> List[Dict]:
    """
    Load test cases from JSON file.
    
    Args:
        json_file_path: Path to JSON file containing test cases
        
    Returns:
        List of test case dictionaries
    """
    with open(json_file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def create_ragas_dataset(test_cases: List[Dict], rag_chain, retriever):
    """
    Convert test cases to RAGAS dataset format.
    Generates responses and contexts dynamically using the RAG chain.
    
    Args:
        test_cases: List of test case dictionaries with user_input, response, retrieved_contexts, reference
        rag_chain: The RAG chain from Lab-10
        retriever: The retriever to get contexts
        
    Returns:
        RAGAS EvaluationDataset ready for evaluation
    """
    user_inputs = []
    responses = []
    retrieved_contexts_list = []
    references = []
    for each_case in tqdm(test_cases):
        user_input = each_case['user_input']
        retrieved_docs = retriever.invoke(user_input)
        
        contexts = [doc.page_content for doc in retrieved_docs]
        response = rag_chain.invoke({
                "input": user_input},
                config = {"configurable": {"session_id": str(uuid.uuid4())}})
        
        answer = response

        user_inputs.append(user_input)
        responses.append(answer)
        retrieved_contexts_list.append(contexts)
        references.append(each_case['reference'])
    
     # Prepare data in the format expected by RAGAS
    data = {
        'user_input': user_inputs,
        'response': responses,
        'retrieved_contexts': retrieved_contexts_list,
        'reference': references
    }

    # Create RAGAS dataset using the correct method

    # JSON into HF dataset format Hugging face format
    dataset = Dataset.from_dict(data) # Convert dictionary to Dataset object
    #HF Datasert -- RAGAS EvaluationDataset format
    return EvaluationDataset.from_hf_dataset(dataset) # Convert Dataset object to EvaluationDataset object


def run_ragas_evaluation(dataset: EvaluationDataset):
    """
    Run RAGAS evaluation on dataset with optional LLM evaluator.
    
    Args:
        dataset: RAGAS EvaluationDataset
        use_llm: Whether to use LLM for evaluation (default: True)
        llm_model: LLM model name if use_llm is True (default: "gpt-5-mini")
        
    Returns:
        EvaluationResult object
    """

    # Define metrics
    metrics = [
        faithfulness,
        answer_relevancy,
        context_precision,
        context_recall,
    ]
    results = evaluate(dataset, metrics=metrics, llm=llm, embeddings=embeddings)
    return results

def main():
    """Main function to run RAG evaluation."""
    print("=" * 60)
    print("RAG Evaluation Lab - SolarNova Dynamics Chatbot")
    print("=" * 60)
    print()

    try:
        # Load test cases from JSON file
        json_file_path = Path(__file__).parent / "rag_dataset.json"
        if not json_file_path.exists():
            print(f"Error: JSON file not found at {json_file_path}")
            print("Please ensure rag_dataset.json exists in the same directory.")
            sys.exit(1)


        test_cases = load_test_cases(str(json_file_path))

        vector_store = get_vector_store()
        rag_chain = create_rag_chain(vector_store)

        retriever = vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k":2}
        )

        dataset = create_ragas_dataset(test_cases, rag_chain, retriever)

        results = run_ragas_evaluation(dataset)

        df = results.to_pandas()
        df.to_csv("ragas_evaluation_results.csv", index=False)

        print("\n" + "=" * 60)
        print("Evaluation Summary Statistics")
        print("=" * 60)
        print()
        
        # Calculate mean scores for each metric
        metrics = ['faithfulness', 'answer_relevancy', 'context_precision', 'context_recall']
        
        for metric in metrics:
            if metric in df.columns:
                mean_score = df[metric].mean()
                std_score = df[metric].std()
                print(f"{metric.replace('_', ' ').title():<25}: {mean_score:.3f} (±{std_score:.3f})")
        
        print()
        print(f"{'Total Questions Evaluated':<25}: {len(df)}")
        print()


    except Exception as e:
        print(f"\nError during evaluation: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


main()