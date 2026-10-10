from anthropic import Anthropic

from dotenv import load_dotenv

load_dotenv()
client = Anthropic()

prompt = "Hello, How are you today?"

response = client.messages.create(model="claude-sonnet-5", messages=[
        {"role": "user", "content": prompt}
    ], max_tokens = 5000)


print(response)
print("--------------------------------")
print("Response:")
print(response.content[0].text)