import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
import dspy

load_dotenv()
API_KEY = os.getenv("GOOGLE_API_KEY")

#chatbot - more stable/predictable
chatbot_llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash", 
    temperature=0.1, 
    google_api_key=API_KEY
)

#customer - more creative/varied
customer_llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash", 
    temperature=0.8, 
    google_api_key=API_KEY
)

#configure dspy to use gemini as the default llm
gemini_lm = dspy.LM(
    model="gemini/gemini-2.5-flash",
    google_api_key=API_KEY,
    cache=False,
    temperature=0.1
)

dspy.configure(lm=gemini_lm)