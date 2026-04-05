import os
from nemoguardrails import RailsConfig, LLMRails
from nemoguardrails.rails.llm.config import Model
import nest_asyncio
nest_asyncio.apply()

# Placeholder that gets replaced per-call
_current_agent = None

async def _agent_action_handler():
    """Global action handler — delegates to whatever agent is currently set."""
    print(f"DEBUG: _agent_action_handler called! _current_agent is None: {_current_agent is None}")
    if _current_agent is None:
        return "I'm not able to help with that request."
    result = await call_agent_action(_current_agent[0], _current_agent[1])
    print(f"DEBUG: action handler got result: '{result[:100]}'")
    return result

def get_nemo_app():
    config = RailsConfig.from_path("./config")
    embedding_model_config = Model(
        type="embeddings",
        engine="google",
        model="gemini-embedding-001"
    )
    config.models.append(embedding_model_config)
    app = LLMRails(config)
    # Register once at startup with the global handler
    app.register_action(_agent_action_handler, name="call_agent_action")
    return app

rails_instance = get_nemo_app()

async def call_agent_action(agent_instance, user_input):
    response = await agent_instance.ainvoke({
        "messages": [("user", user_input)],
        "test_type": "p1",
        "next_agent": "chatbot",
        "chat_finished": False,
        "api_calls": []
    })
    return response["messages"][-1].content

async def run_guarded_agent(agent, user_input: str):
    global _current_agent
    _current_agent = (agent, user_input)

    result = await rails_instance.generate_async(messages=[{
        "role": "user",
        "content": user_input
    }])

    content = result.get("content", "").strip()
    _current_agent = None

    if not content:
        # NeMo returned empty — call agent directly as fallback
        print("DEBUG NEMO: Empty response, using direct agent fallback")
        content = await call_agent_action(agent, user_input)
    else:
        print(f"DEBUG NEMO: Rails responded: '{content[:80]}'")

    return content