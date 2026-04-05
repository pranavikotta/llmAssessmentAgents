# graphs.py
import json
import re
from typing import TypedDict, List, Optional
from typing_extensions import Annotated
import yaml

from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, END
from langchain_core.messages import SystemMessage, AIMessage

from agent import chatbot_llm, customer_llm, dspy_lm
from prompt_optimization import AIAssistant

assistant = AIAssistant()  # initializes dspy module

class ChatTesterState(TypedDict):
    messages: Annotated[list, add_messages]  # reducer: append not overwrite
    api_calls: Optional[List[str]]
    test_type: str
    next_agent: str   # customer or chatbot
    chat_finished: bool

with open("customer_personas.yaml", "r", encoding="utf-8") as f:
    persona_config = yaml.safe_load(f)

def customer_node(state: ChatTesterState) -> ChatTesterState:
    messages = state["messages"]
    test_id = state["test_type"]
    persona_data = persona_config["personas"].get(test_id)
    if not persona_data:
        raise ValueError(f"Persona {test_id} not found.")
    persona = SystemMessage(content=persona_data["system_prompt"])
    config = {"configurable": {"temperature": persona_data.get("temp", 0.7)}}
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
    try:
        data = json.loads(content)
        return [data]
    except json.JSONDecodeError:
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
    history_str = "\n".join([f"{m.type}: {m.content}" for m in messages[:-1]])

    check_query = f"Does this message contain a specific location? Answer Yes or No: {last_msg}"
    intent_result = chatbot_llm.invoke(check_query).content.lower()
    location_detected = "yes" in intent_result

    if location_detected:
        print("Mode: Recommendations (JSON)")
        catalogue_data = "Maxar-15cm|pid:644b|oid:0134\nICEYE-SAR|pid:992a|oid:0882"

        # Fixed: unpack Prediction.answer instead of treating as tuple
        dspy_output = assistant.forward(
            question=last_msg,
            history=history_str,
            product_catalogue=catalogue_data,
            location_detected=True
        )
        content_output = dspy_output.answer
        internal_reasoning = None
    else:
        print("Mode: General QA (Text)")

        # Fixed: unpack Prediction fields instead of treating as tuple
        result = assistant.forward(
            question=last_msg,
            history=history_str,
            location_detected=False
        )
        content_output = result.answer
        internal_reasoning = result.reasoning if hasattr(result, 'reasoning') else None

        if internal_reasoning:
            print(f"DEBUG REASONING: {internal_reasoning}")

    response = AIMessage(content=content_output)
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
    """Full simulation loop — customer node vs chatbot node.
    Used for the LangGraph self-contained simulation tests."""
    wf = StateGraph(ChatTesterState)
    wf.add_node("customer", customer_node)
    wf.add_node("chatbot", chatbot_node)

    wf.add_conditional_edges(
        "customer",
        route_logic,
        {"chatbot": "chatbot", "END": END}
    )
    wf.add_conditional_edges(
        "chatbot",
        route_logic,
        {"customer": "customer", "END": END}
    )

    wf.set_entry_point("customer")
    return wf.compile()


def chatbot_only_workflow():
    """Single-node workflow for external callers like PyRIT and NeMo.
    Skips customer_node entirely — PyRIT acts as the attacker instead."""
    wf = StateGraph(ChatTesterState)
    wf.add_node("chatbot", chatbot_node)
    wf.add_conditional_edges(
        "chatbot",
        route_logic,
        {"customer": END, "END": END}
    )
    wf.set_entry_point("chatbot")
    return wf.compile()