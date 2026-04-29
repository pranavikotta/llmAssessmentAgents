import asyncio
import json
import re
import os
from pyrit.prompt_target import PromptTarget
from pyrit.models import Message
from nemo_defense import run_guarded_agent

# Set DEBUG_JSON_EXTRACTION=1 in .env to see extraction details.
# Off by default to keep terminal readable during normal runs.
_DEBUG_EXTRACTION = os.getenv("DEBUG_JSON_EXTRACTION", "0") == "1"

_FALLBACK_PLAINTEXT = (
    "I can help with satellite imagery recommendations. "
    "Could you provide a specific location or area of interest "
    "so I can suggest the best options for you?"
)


def _json_to_plaintext(content: str) -> str:
    """
    Converts a recommender JSON blob to readable plaintext so PyRIT's
    RedTeamingAttack can continue the conversation.

    Handles all known failure modes that caused 'Strategy execution failed':
      1. DSPy wraps output under 'recommendation_json' key instead of unwrapping
      2. DSPy outputs 'recommendation_json: {...}' as a prefixed string
      3. Empty dict {} with no extractable fields
      4. response field missing or empty
      5. Completely unparseable content

    Always returns a non-empty string — worst case returns _FALLBACK_PLAINTEXT.
    DEBUG_JSON_EXTRACTION=1 enables verbose output for debugging new failure modes.
    """
    if not content or not content.strip():
        if _DEBUG_EXTRACTION:
            print("DEBUG extraction: empty content, using fallback")
        return _FALLBACK_PLAINTEXT

    raw = content.strip()

    # ── Step 1: try to get a dict from the content ────────────────────────────
    data = None

    try:
        parsed = json.loads(raw)
        if isinstance(parsed, dict):
            data = parsed
    except json.JSONDecodeError:
        pass

    # DSPy prefix pattern: 'recommendation_json: {...}'
    if data is None:
        prefix_match = re.match(
            r'recommendation_json\s*:\s*(\{.*\})', raw, re.DOTALL | re.IGNORECASE
        )
        if prefix_match:
            try:
                data = json.loads(prefix_match.group(1))
            except json.JSONDecodeError:
                pass

    # Grab first {...} block from anywhere in the string
    if data is None:
        block_match = re.search(r'\{.*\}', raw, re.DOTALL)
        if block_match:
            try:
                data = json.loads(block_match.group(0))
            except json.JSONDecodeError:
                pass

    if data is None:
        if _DEBUG_EXTRACTION:
            print(f"DEBUG extraction: could not parse JSON. Content[:120]: {raw[:120]}")
        # Not JSON at all — return as-is (plain text QA response)
        return content

    # ── Step 2: unwrap DSPy nested wrapper if present ─────────────────────────
    if "recommendation_json" in data and isinstance(data["recommendation_json"], dict):
        data = data["recommendation_json"]
    elif "recommendation_json" in data and isinstance(data["recommendation_json"], str):
        try:
            inner = json.loads(data["recommendation_json"])
            if isinstance(inner, dict):
                data = inner
        except json.JSONDecodeError:
            pass

    # ── Step 3: extract readable parts ───────────────────────────────────────
    parts = []

    response_text = data.get("response") or data.get("Response") or ""
    if response_text and response_text.strip():
        parts.append(response_text.strip())

    products = data.get("suggestedProducts") or data.get("suggested_products") or []
    if products and isinstance(products, list):
        names = [
            p.get("name", "") for p in products
            if isinstance(p, dict) and p.get("name")
        ]
        if names:
            # Cap at 3 product names — long lists cause the attacker LLM to mimic
            # JSON structure in its reply, crashing PyRIT's JSONAdapter.
            display_names = names[:3]
            suffix = f" (and {len(names)-3} more)" if len(names) > 3 else ""
            parts.append(f"Suggested products: {', '.join(display_names)}{suffix}.")

    needs_clarif = (
        data.get("needsLocationClarification")
        or data.get("needs_location_clarification")
    )
    if needs_clarif:
        parts.append(
            "Could you provide a specific location so I can refine these recommendations?"
        )

    if parts:
        result = " ".join(parts)
        # Hard cap: prevents attacker LLM seeing wall-of-text that triggers JSON mimicry.
        if len(result) > 400:
            result = result[:397] + "..."
        if _DEBUG_EXTRACTION:
            print(f"DEBUG extraction: extracted {len(parts)} parts. Result[:80]: {result[:80]}")
        return result

    # ── Step 4: last resort — use first non-empty string value ────────────────
    str_values = [v for v in data.values() if isinstance(v, str) and v.strip()]
    if str_values:
        result = str_values[0].strip()
        if _DEBUG_EXTRACTION:
            print(f"DEBUG extraction: used first string value. Result[:80]: {result[:80]}")
        return result

    if _DEBUG_EXTRACTION:
        print(f"DEBUG extraction: dict has no usable fields. Keys: {list(data.keys())}")
    return _FALLBACK_PLAINTEXT


async def _invoke_agent(agent_instance, user_input: str):
    """
    Directly invokes the LangGraph agent without NeMo.
    Returns (answer, turn_log) so the caller can log per-turn data.
    turn_log is populated by chatbot_node via TurnLogger.record().
    """
    response = await agent_instance.ainvoke({
        "messages": [("user", user_input)],
        "test_type": "p1",
        "next_agent": "chatbot",
        "chat_finished": False,
        "api_calls": [],
        "turn_log": [],
    })
    return response["messages"][-1].content, response.get("turn_log", [])


class AgentTarget(PromptTarget):
    def __init__(self, agent, use_nemo: bool = True):
        super().__init__()
        self.agent = agent
        self.system_prompt = None
        self.use_nemo = use_nemo
        # Stores turn_log from last call — read by run_condition after each round
        self.last_turn_log = []

    def set_system_prompt(self, system_prompt: str, **kwargs):
        self.system_prompt = system_prompt

    def _validate_request(self, *, message: Message) -> None:
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
            self.last_turn_log = []
        else:
            answer, self.last_turn_log = await _invoke_agent(self.agent, user_input)

        answer = _json_to_plaintext(answer)

        msg = Message.from_prompt(prompt=answer, role="assistant")
        msg.message_pieces[0].converted_value = answer
        msg.message_pieces[0].converted_value_data_type = "text"
        return [msg]

    def is_json_response_supported(self) -> bool:
        return False


print("AgentWrapper loaded successfully for PyRIT 0.11.0")