from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
import os
load_dotenv()

from litellm import cost_per_token

openai_llm = init_chat_model("gpt-4o-mini", model_provider="openai")
gemini_llm = init_chat_model("gemini-2.5-flash", model_provider="google_genai")
anthropic_llm = init_chat_model("claude-haiku-4-5-20251001", model_provider="anthropic")

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant."),
    ("user", "Explain quantum computing in 10 sentences."),
])

chain_openai = prompt | openai_llm
chain_gemini = prompt | gemini_llm
# chain_anthropic = prompt | anthropic_llm

def print_usage(response, provider_name, llm_name):
    usage = response.usage_metadata
    input_tokens = usage["input_tokens"]
    output_tokens = usage["output_tokens"]
    total_tokens = usage["total_tokens"]
    print(f"Input Tokens:  {input_tokens}")
    print(f"Output Tokens: {output_tokens}")
    print(f"Total Tokens:  {total_tokens}")

    input_cost, output_cost = cost_per_token(
            model=f"{provider_name}/{llm_name}",
            prompt_tokens=input_tokens,
            completion_tokens=output_tokens,
        )
    total_cost = float(input_cost + output_cost)
    print(f"Input Cost:   ${input_cost:.6f}")
    print(f"Output Cost:  ${output_cost:.6f}")
    print(f"Total Cost:   ${total_cost:.6f}")



res_openai = chain_openai.invoke({}) 
print("OpenAI LLM Usage and Cost:")
print_usage(res_openai, "openai", "gpt-4o-mini")
print("\n\n")
res_gemini = chain_gemini.invoke({}) 
print("Gemini LLM Usage and Cost:")
print_usage(res_gemini, "gemini", "gemini-2.5-flash")

llm_groq = init_chat_model("openai/gpt-oss-120b", 
                      model_provider="openai",
                      base_url="https://api.groq.com/openai/v1",
                      api_key = os.getenv("GROQ_API_KEY"))
chain_groq = prompt | llm_groq

res_groq = chain_groq.invoke({}) 
print("Groq LLM Usage and Cost:")
print_usage(res_groq, "groq", "openai/gpt-oss-120b")
print("\n\n")