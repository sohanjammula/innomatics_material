from google import genai
from dotenv import load_dotenv  # To load the .env file

load_dotenv()  # Load the environment variables from the .env file

#Initialize the Gemini API with the API key
client = genai.Client()


prompt = '''
In just 2 lines explain about AI?

'''

result = client.models.generate_content(model = "gemini-2.5-flash-lite", contents = prompt)


print(result)
print("------------------------------------")

print(result.text)