import os
import torch
import accelerate
from dotenv import load_dotenv
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain_huggingface import HuggingFacePipeline, ChatHuggingFace

load_dotenv()
os.environ['HF_TOKEN'] = os.getenv('HF_Access_Token')
os.environ['HUGGINGFACEHUB_API_TOKEN'] = os.getenv('HF_Access_Token')

# 1. Download model
print("Downloading DeepSeek Coder...")
model_id = "deepseek-ai/deepseek-coder-1.3b-instruct"
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True)

# 2. Save to SINGLE CORRECT FOLDER (raw strings, forward slashes)
SAVE_PATH = r"C:/Users/DELL/Music/deepseek-coder-1.3b"

print(f"Saving to {SAVE_PATH}...")
tokenizer.save_pretrained(SAVE_PATH)
model.save_pretrained(SAVE_PATH)

# 3. Load from SAME PATH with local_files_only
print("Loading saved model...")
saved_tokenizer = AutoTokenizer.from_pretrained(
    SAVE_PATH, 
    local_files_only=True,
    trust_remote_code=True
)
saved_model = AutoModelForCausalLM.from_pretrained(
    SAVE_PATH,
    torch_dtype=torch.float16,
    device_map="auto",
    local_files_only=True, 
    trust_remote_code=True
)

# 4. Create pipeline
pipe = pipeline(
    "text-generation",
    model=saved_model,
    tokenizer=saved_tokenizer,
    max_new_tokens=256,
    temperature=0.1,
    do_sample=True
)

llm = HuggingFacePipeline(pipeline=pipe)
print("Pipeline ready!")

# Test
response = llm.invoke("Write Python fibonacci function:")
print(response)

model = ChatHuggingFace(llm = llm)
print(model)

# Prompts/Message
messages = [
    ('system', 'Generate code and example inputs and outputs.'),
    ('human', 'Create a class to train a ml model')
]

# Invoke/Respose
response = model.invoke(messages)
print(response.content)