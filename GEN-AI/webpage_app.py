import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
import streamlit as st
import zipfile

load_dotenv()
os.environ['GOOGLE_API_KEY'] = os.getenv('gemini_key')

st.title('Automatic Webpage Creation')

model = ChatGoogleGenerativeAI(
    model = 'gemini-2.5-flash')

description = st.text_area('Describe the type of webpage you want to create')
content = st.text_area('Post the content for webpage creation')

if st.button('Generate'):
    system_template = """You are a Senior Frontend Web Developer with 10+ years experience in HTML5, CSS3, and modern JavaScript (ES6+).

    Your task: Generate COMPLETE, PRODUCTION-READY frontend code based on user requirements.

    **MANDATORY OUTPUT FORMAT** (exact delimiters):
    --html--
    [html code here]
    --html--

    --css--
    [css code here]
    --css--

    --js--
    [java script code here]
    --js--
    """
    human_template = "Build a {description} using following {content}"

    system_message = SystemMessagePromptTemplate.from_template(system_template)
    human_message = HumanMessagePromptTemplate.from_template(human_template)

    web_dev_template = ChatPromptTemplate.from_messages([system_message, human_message])
    prompt = web_dev_template.invoke({'description' : description, 'content': content})
    response = model.invoke(prompt)

    with open('index.html', 'w') as file:
            file.write(response.content.split('--html--')[1])

    with open('style.css', 'w') as file:
        file.write(response.content.split('--css--')[1])

    with open('script.js', 'w') as file:
        file.write(response.content.split('--js--')[1])

    with zipfile.ZipFile('website.zip', 'w') as zip:
        zip.write('index.html')
        zip.write('style.css')
        zip.write('script.js')

    st.download_button('click here to download', data = open('website.zip', 'rb'), 
                        file_name = 'website.zip')

    st.write('Success')

    # Dynamic 5 multiple choice questions assessment webpage with result and pdf downloadable report