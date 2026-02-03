import os
from nemoguardrails import RailsConfig, LLMRails

#initialize rails from the config directory
config = RailsConfig.from_path("./config")
rails = LLMRails(config)

async def run_guarded_agent(agent, user_input: str):
    #nemo processes input, agent logic, and output
    #define an action that calls your langgraph agent
    async def call_agent_action(context=None):
        response = await agent.ainvoke({"question": user_input})
        return response.get("answer", "no response")

    #register the agent as a custom action for nemo
    rails.register_action(call_agent_action, name="call_agent_action")

    #execute with rails
    #nemo will check input -> run agent -> check output
    result = await rails.generate_async(prompt=user_input)
    
    return result