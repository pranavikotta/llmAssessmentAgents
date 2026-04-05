import os
from dotenv import load_dotenv
import dspy

load_dotenv()

# ── PROVIDER CONFIG ───────────────────────────────────────────────────────────
# Set LLM_PROVIDER=grok or LLM_PROVIDER=gemini in .env file
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "gemini").lower()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GROK_API_KEY = os.getenv("GROK_API_KEY")

# ── LLM INITIALISATION ────────────────────────────────────────────────────────
if LLM_PROVIDER == "grok":
    from langchain_openai import ChatOpenAI

    # Grok uses OpenAI-compatible API — flash non-reasoning model
    chatbot_llm = ChatOpenAI(
        model="grok-4-1-fast-non-reasoning",
        temperature=0.1,
        api_key=GROK_API_KEY,
        base_url="https://api.x.ai/v1",
    )

    customer_llm = ChatOpenAI(
        model="grok-4-1-fast-non-reasoning",
        temperature=0.8,
        api_key=GROK_API_KEY,
        base_url="https://api.x.ai/v1",
    )

    # DSPy also supports OpenAI-compatible endpoints
    dspy_lm = dspy.LM(
        model="openai/grok-4-1-fast-non-reasoning",
        api_key=GROK_API_KEY,
        api_base="https://api.x.ai/v1",
        cache=False,
        temperature=0.1,
    )

else:
    # Default: Gemini
    from langchain_google_genai import ChatGoogleGenerativeAI

    chatbot_llm = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash",
        temperature=0.1,
        google_api_key=GOOGLE_API_KEY,
    )

    customer_llm = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash",
        temperature=0.8,
        google_api_key=GOOGLE_API_KEY,
    )

    dspy_lm = dspy.LM(
        model="gemini/gemini-2.5-flash",
        google_api_key=GOOGLE_API_KEY,
        cache=False,
        temperature=0.1,
    )

dspy.configure(lm=dspy_lm)
print(f"agent.py loaded — provider: {LLM_PROVIDER.upper()}")