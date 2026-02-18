#creating two agents (customer and chatbot) that use the same LLM to have a conversation
import json
import re
from typing import TypedDict, List, Optional
from typing_extensions import Annotated
import yaml

from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, END  # Import END here
from langchain_core.messages import SystemMessage, AIMessage

from agent import chatbot_llm, customer_llm, gemini_lm
from prompt_optimization import AIAssistant

assistant = AIAssistant() #initializes dspy module

class ChatTesterState(TypedDict):
    messages: Annotated[list, add_messages] #reducer: append not overwrite
    api_calls: Optional[List[str]]
    test_type: str
    next_agent: str #customer or chatbot
    chat_finished: bool #whether chat ended or not

with open("customer_personas.yaml", "r", encoding="utf-8") as f:
    persona_config = yaml.safe_load(f)

def customer_node(state: ChatTesterState) -> ChatTesterState:
    messages = state["messages"]
    test_id = state["test_type"]
    
    #load system prompt (persona data)
    persona_data = persona_config["personas"].get(test_id)
    if not persona_data:
        raise ValueError(f"Persona {test_id} not found.")
    persona = SystemMessage(content=persona_data["system_prompt"])
    
    #use the specific temperature from YAML if available
    config = {"configurable": {"temperature": persona_data.get("temp", 0.7)}} #default temp 0.7 if not specified
    response = customer_llm.invoke([persona] + messages, config=config)
    
    return {
        "messages": [response], 
        "next_agent": "chatbot", 
        "chat_finished": False, 
        "test_type": test_id, 
        "api_calls": state.get("api_calls") or []
    }

def check_for_json(response):
    content = response.content.strip()
    #chatbot system prompt mandates the entire response be JSON, not contain a block
    #try a direct parse first to test strict compliance
    try:
        data = json.loads(content)
        return [data] #success: The agent followed the 'no-markdown' rule
    except json.JSONDecodeError:
        #extraction logic if the agent failed the "No Markdown" rule but still produced JSON inside backticks
        match = re.search(r"\{.*\}", content, re.DOTALL)
        if match:
            try:
                return [json.loads(match.group(0))]
            except:
                return []
    return []

with open("chatbot_system_prompts.yaml", "r", encoding="utf-8") as f:
    system_prompt_config = yaml.safe_load(f)

def chatbot_node(state: ChatTesterState) -> ChatTesterState:
    messages = state["messages"]
    last_msg = messages[-1].content

    #format history for dspy module
    history_str = "\n".join([f"{m.type}: {m.content}" for m in messages[:-1]])
    
    #intent detection: use the LLM to decide if the user mentioned a location, determine chatbot mode
    check_query = f"Does this message contain a specific location? Answer Yes or No: {last_msg}"
    intent_result = chatbot_llm.invoke(check_query).content.lower()
    location_detected = "yes" in intent_result

    #select prompt form yaml based on detected intent, execute with dspy assistant
    if location_detected:
        print("Mode: Recommendations (JSON)")
        catalogue_data = "Maxar-15cm|pid:644b|oid:0134\nICEYE-SAR|pid:992a|oid:0882" #simulated catalogue data

        #recommendation response object is returned
        dspy_output = assistant.forward(
            question=last_msg, 
            history=history_str, 
            product_catalogue=catalogue_data, 
            location_detected=True
        )
        content_output = dspy_output.model_dump_json(by_alias=True) #converts pydantic object to json    
        internal_reasoning = None
    else:
        #stay in conversational mode
        print("Mode: General QA (Text)")
        content_output, internal_reasoning = assistant.forward(
            question=last_msg, 
            history=history_str, 
            location_detected=False
        )
        if internal_reasoning:
            print(f"DEBUG REASONING: {internal_reasoning}")

    #wrap result in langchain message for graph
    response = AIMessage(content=content_output)
    
    #extract and update state, check if any JSON API calls were made
    new_extracted = check_for_json(response)
    existing_calls = state.get("api_calls") or []
    
    return {
        "messages": [response], 
        "next_agent": "customer", 
        "chat_finished": False, 
        "test_type": state["test_type"], 
        "api_calls": existing_calls + new_extracted
    }

def route_logic(state: ChatTesterState) -> str:
    if "test_complete" in state["messages"][-1].content.lower():
        return "END"
    if len(state["messages"]) >= 15:
        print("Max turns reached, ending chat.")
        return "END"
    return state["next_agent"]

def workflow():
    workflow = StateGraph(ChatTesterState)
    workflow.add_node("customer", customer_node)
    workflow.add_node("chatbot", chatbot_node)
    
    # Conditional edges - map "END" string to END constant
    workflow.add_conditional_edges(
        "customer", 
        route_logic, 
        {"chatbot": "chatbot", "END": END}  # Use END constant here
    )
    workflow.add_conditional_edges(
        "chatbot", 
        route_logic, 
        {"customer": "customer", "END": END}  # Use END constant here
    )

    workflow.set_entry_point("customer")
    return workflow.compile()