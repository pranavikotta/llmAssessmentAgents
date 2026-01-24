import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
import dspy

load_dotenv()
API_KEY = os.getenv("API_KEY")

#chatbot - more stable/predictable
chatbot_llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash", 
    temperature=0.1, 
    api_key=API_KEY
)

#customer - more creative/varied
customer_llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash", 
    temperature=0.8, 
    api_key=API_KEY
)

#configure dspy to use gemini as the default llm
gemini_lm = dspy.LM(
    model="gemini-2.5-flash",
    api_key=API_KEY,
    temperature=0.1
)

dspy.configure(lm=gemini_lm)