import os
from nemoguardrails import RailsConfig, LLMRails
from nemoguardrails.rails.llm.config import Model
import nest_asyncio
nest_asyncio.apply()

# nemo_defense.py

def get_nemo_app():
    config = RailsConfig.from_path("./config")
    
    # Switch engine to "google" (standard) and remove the prefix
    # NeMo's 'google' provider uses the vertex/generative AI SDK differently 
    # than 'google_genai'
    embedding_model_config = Model(
        type="embeddings",
        engine="google", 
        model="gemini-embedding-001" 
    )
    
    config.models.append(embedding_model_config)
    app = LLMRails(config)
    return app

# Initialize the global instance that agent_wrapper.py uses
rails_instance = get_nemo_app()

async def call_agent_action(agent_instance, user_input):
    # LangGraph returns a dict; access the content of the last message
    response = await agent_instance.ainvoke({"messages": [("user", user_input)]})
    return response["messages"][-1].content

async def run_guarded_agent(agent, user_input: str):
    async def wrapped_action():
        return await call_agent_action(agent, user_input)

    # Register/Override action for this session
    rails_instance.register_action(wrapped_action, name="call_agent_action")

    result = await rails_instance.generate_async(messages=[{
        "role": "user", 
        "content": user_input
    }])
    
    return result["content"]