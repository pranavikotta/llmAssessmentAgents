# agent_wrapper.py
import asyncio
# dynamic import to handle pyrit versioning differences
try:
    from pyrit.models import PromptRequestResponse as RequestType, PromptResponse
except ImportError:
    # fallback for older versions
    from pyrit.models import PromptRequestPiece as RequestType, PromptResponse

from pyrit.prompt_target import PromptTarget
from nemo_defense import run_guarded_agent

class AgentTarget(PromptTarget):
    def __init__(self, agent):
        # initialize with langgraph agent
        super().__init__()
        self.agent = agent
    
    async def send_prompt_async(self, *, prompt_request: RequestType) -> PromptResponse:
        # extract text from pieces if container, else direct attribute
        if hasattr(prompt_request, "request_pieces") and prompt_request.request_pieces:
            user_input = prompt_request.request_pieces[0].converted_value
        else:
            user_input = getattr(prompt_request, "converted_value", str(prompt_request))

        # route through nemo guardrails defense layer
        answer = await run_guarded_agent(self.agent, user_input)

        return PromptResponse(text=answer)