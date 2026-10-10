from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableBranch, RunnableLambda, RunnablePassthrough
from langchain.chat_models import init_chat_model
from dotenv import load_dotenv

load_dotenv()
import os

llm = init_chat_model("openai/gpt-oss-120b", 
                      model_provider="openai",
                      base_url="https://api.groq.com/openai/v1",
                      api_key = os.getenv("GROQ_API_KEY"))

prompt = PromptTemplate.from_template(
    """You are an accounts payable assistant.

Vendor: {vendor}
Items: {items}
Invoice total: ${total:.2f}

Write a one-paragraph note for the finance team summarizing this invoice."""
)

summary_chain = prompt | llm | StrOutputParser()

def calculate_total(input: dict) -> float:
    return sum(item["quantity"] * item["unit_price"] for item in input["items"])

office_supplies = {
    "vendor": "Acme Office Supplies",
    "items": [
        {"description": "Printer paper", "quantity": 10, "unit_price": 8},
        {"description": "Pens (box)", "quantity": 4, "unit_price": 5},
    ],
}

# new_laptops = {
#     "vendor": "Acme IT Hardware",
#     "items": [
#         {"description": "Laptop", "quantity": 3, "unit_price": 950},
#     ],
# }
#RunnableLambda wraps a plain Python function into langchain's Runnable interface, allowing it to be used in a chain of operations. In this case, it calculates the total cost of the invoice items.
# final_chain = RunnablePassthrough.assign(total=RunnableLambda(calculate_total)) | summary_chain

# response = final_chain.invoke(office_supplies)

AUTO_APPROVE_LIMIT = 1000
MANAGER_APPROVE_LIMIT = 5000

auto_approve_prompt = PromptTemplate.from_template(
    "Write a one-line note confirming this ${total:.2f} invoice from "
    "{vendor} is auto-approved -- no manager sign-off needed."
)
manager_approve_prompt = PromptTemplate.from_template(
    "Write a short note asking a manager to review and approve this "
    "${total:.2f} invoice from {vendor}."
)
finance_review_prompt = PromptTemplate.from_template(
    "Write a short note flagging this ${total:.2f} invoice from {vendor} "
    "for finance director review -- it's a large purchase."
)

auto_approve_chain = auto_approve_prompt | llm | StrOutputParser()
manager_approve_chain = manager_approve_prompt | llm | StrOutputParser()
finance_review_chain = finance_review_prompt | llm | StrOutputParser() 

routing_chain = RunnableBranch(
    (lambda input: input["total"] <= AUTO_APPROVE_LIMIT, auto_approve_chain),
    (lambda input: input["total"] <= MANAGER_APPROVE_LIMIT, manager_approve_chain),
    finance_review_chain
)

invoice_chain = RunnablePassthrough.assign(total=RunnableLambda(calculate_total)) | routing_chain

office_supplies = {
    "vendor": "Acme Office Supplies",
    "items": [{"description": "Printer paper", "quantity": 10, "unit_price": 8}],
}
new_laptops = {
    "vendor": "Acme IT Hardware",
    "items": [{"description": "Laptop", "quantity": 3, "unit_price": 950}],
}
office_renovation = {
    "vendor": "Acme Contractors",
    "items": [{"description": "Renovation", "quantity": 1, "unit_price": 12000}],
}


response = invoice_chain.batch([office_supplies, new_laptops, office_renovation])

# print(response)

for i, each_response in enumerate(response):
    print(f"Invoice {i+1} response: {each_response}")