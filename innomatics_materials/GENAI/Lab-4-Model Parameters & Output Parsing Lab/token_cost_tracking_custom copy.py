from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
import json

load_dotenv()

openai_llm = init_chat_model("gpt-4o-mini", model_provider="openai")
gemini_llm = init_chat_model("gemini-2.0-flash", model_provider="google_genai")
anthropic_llm = init_chat_model("claude-haiku-4-5-20251001", model_provider="anthropic")


prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant."),
        ("user", "Explain quantum computing in 10 sentences.")
    ])
# Pricing per million tokens
PRICING = {
    "openai": {
        "gpt-4o-mini": {
            "input": 0.15,   # $0.15 per million input tokens
            "output": 0.60   # $0.60 per million output tokens
        }
    },
    "google_genai": {
        "gemini-2.0-flash": {
            "input": 0.075,  # $0.075 per million input tokens
            "output": 0.30   # $0.30 per million output tokens
        }
    },
    "anthropic": {
        "claude-haiku-4-5-20251001": {
            "input": 0.25,   # $0.25 per million input tokens
            "output": 1.25   # $1.25 per million output tokens
        }
    }
}

def calculate_cost(input_tokens, output_tokens, provider_name, model_name):
    """Calculate cost based on token usage and provider pricing."""
    provider_key = None
    if provider_name.lower() == "openai":
        provider_key = "openai"
    elif "gemini" in provider_name.lower() or "google" in provider_name.lower():
        provider_key = "google_genai"
    elif "anthropic" in provider_name.lower() or "claude" in provider_name.lower():
        provider_key = "anthropic"
    
    if not provider_key or provider_key not in PRICING:
        return None, None, None
    
    # Find matching model in pricing
    model_pricing = None
    for model_key, pricing in PRICING[provider_key].items():
        if model_key in model_name.lower() or model_name.lower() in model_key:
            model_pricing = pricing
            break
    
    if not model_pricing:
        return None, None, None
    
    # Calculate costs
    input_cost = (input_tokens / 1_000_000) * model_pricing["input"]
    output_cost = (output_tokens / 1_000_000) * model_pricing["output"]
    total_cost = input_cost + output_cost
    
    return input_cost, output_cost, total_cost

def print_token_usage(response, provider_name, model_name):
    """Helper function to extract and print token usage information."""
    print(f"\n{'='*60}")
    print(f"Token Usage for {provider_name}")
    print(f"{'='*60}")
    if hasattr(response, 'usage_metadata') and response.usage_metadata:
        usage = response.usage_metadata
        print(f"Input Tokens:  {usage.get('input_tokens', 'N/A')}")
        print(f"Output Tokens: {usage.get('output_tokens', 'N/A')}")
        print(f"Total Tokens:  {usage.get('total_tokens', 'N/A')}")

        input_cost, output_cost, total_cost = calculate_cost(usage.get('input_tokens', 0), usage.get('output_tokens', 0), provider_name, model_name)
        print(f"Input Cost: {round(input_cost, 4)}")
        print(f"Output Cost: {round(output_cost, 4)}")
        print(f"Total Cost: {round(total_cost, 4)}")

       
try:
    chain = prompt | openai_llm
    openai_response = chain.invoke({})
    
    print(f"\nResponse: {openai_response.content}...")
    print_token_usage(openai_response, "openai", "gpt-4o-mini")
    
except Exception as e:
    print(f"Error with OpenAI: {e}")

try:
    chain = prompt | gemini_llm
    gemini_response = chain.invoke({})
    print(f"\nResponse: {gemini_response.content[:100]}...")
    print_token_usage(gemini_response, "google_genai", "gemini-2.0-flash")
except Exception as e:
    print(f"Error with Gemini: {e}")

try:
    chain = prompt | anthropic_llm
    anthropic_response = chain.invoke({})
    print(f"\nResponse: {anthropic_response.content[:100]}...")
    print_token_usage(anthropic_response, "anthropic", "claude-haiku-4-5-20251001")
except Exception as e:
    print(f"Error with Anthropic: {e}")



