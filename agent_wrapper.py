import asyncio
import json
import re
from pyrit.prompt_target import PromptTarget
from pyrit.models import Message
from nemo_defense import run_guarded_agent, call_agent_action


def _json_to_plaintext(content: str) -> str:
    """
    If the agent returned a JSON recommendations blob, extract the human-readable
    'response' field and product names so PyRIT can continue the conversation.
    Without this, PyRIT's RedTeamingAttack throws 'Strategy execution failed'
    when it receives a raw JSON string as a chat turn (P2, P3, P6 personas).
    """
    try:
        # Try full parse first
        data = json.loads(content)
    except json.JSONDecodeError:
        # Try extracting a JSON object from within the string
        match = re.search(r"\{.*\}", content, re.DOTALL)
        if not match:
            return content  # not JSON, return as-is
        try:
            data = json.loads(match.group(0))
        except json.JSONDecodeError:
            return content

    parts = []
    if isinstance(data, dict):
        if "response" in data:
            parts.append(data["response"])
        products = data.get("suggestedProducts", [])
        if products:
            names = [p.get("name", "") for p in products if p.get("name")]
            if names:
                parts.append(f"Suggested products: {', '.join(names)}.")
        if data.get("needsLocationClarification"):
            parts.append("Could you provide a specific location so I can refine these recommendations?")
    return " ".join(parts) if parts else content

class AgentTarget(PromptTarget):
    def __init__(self, agent, use_nemo: bool = True):
        super().__init__()
        self.agent = agent
        self.system_prompt = None
        self.use_nemo = use_nemo  # Whether to route through NeMo Guardrails

    def set_system_prompt(self, system_prompt: str, **kwargs):
        """
        Required by PyRIT 0.11.0 - sets the system prompt for the agent.
        LangGraph agent has its own system prompts from YAML file,
        this is for PyRIT's internal tracking only.
        """
        self.system_prompt = system_prompt

    def _validate_request(self, *, message: Message) -> None:
        """Validate the message request"""
        if not message:
            raise ValueError("Message is empty")
        try:
            content = message.get_value()
            if not content:
                raise ValueError("Message content is empty")
        except Exception as e:
            raise ValueError(f"Invalid message: {e}")

    async def send_prompt_async(self, *, message: Message) -> list[Message]:
        user_input = message.get_value()

        if self.use_nemo:
            answer = await run_guarded_agent(self.agent, user_input)
        else:
            answer = await call_agent_action(self.agent, user_input)

        # If the agent returned a JSON recommendations blob, convert it to
        # readable plaintext so PyRIT's RedTeamingAttack can continue the
        # conversation. Raw JSON causes 'Strategy execution failed' errors
        # for personas that trigger Recommendations mode (P2, P3, P6).
        answer = _json_to_plaintext(answer)

        msg = Message.from_prompt(prompt=answer, role="assistant")
        msg.message_pieces[0].converted_value = answer
        msg.message_pieces[0].converted_value_data_type = "text"
        return [msg]

    def is_json_response_supported(self) -> bool:
        """
        Indicates whether the target supports JSON response format.
        Required abstract method from PromptTarget.
        Returns False - the agent returns conversational text, not raw JSON.
        """
        return False

print("AgentWrapper loaded successfully for PyRIT 0.11.0")