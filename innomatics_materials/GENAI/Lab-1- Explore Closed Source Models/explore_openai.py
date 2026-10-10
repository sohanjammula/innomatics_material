from openai import OpenAI 

from dotenv import load_dotenv

load_dotenv()


client = OpenAI()

prompt = "Hi, How are you?"

#Openai Prompt Structure

#[{"role":"user","content":"Hi, How are you?"}]

result = client.chat.completions.create(model = "gpt-4o",
                                messages = [{"role":"user","content":prompt}])

print(result)

print("------------------------------------")

print(result.choices[0].message.content)