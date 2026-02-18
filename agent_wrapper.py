import asyncio
from pyrit.prompt_target import PromptTarget
from pyrit.models import Message
from nemo_defense import run_guarded_agent

class AgentTarget(PromptTarget):
    def __init__(self, agent):
        super().__init__()
        self.agent = agent
    
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
        """
        Processes a prompt by routing it through NeMo Guardrails and LangGraph.
        
        Args:
            message: Message object containing the user input
            
        Returns:
            List containing a single Message object with the assistant's response
        """
        # Extract the text from the Message object
        user_input = message.get_value()

        # Route through NeMo Guardrails defense layer
        # This calls your LangGraph agent internally via the action registered in nemo_defense
        answer = await run_guarded_agent(self.agent, user_input)

        # Create a response Message using from_prompt helper
        # CORRECT parameters: prompt (not content), role
        response_message = Message.from_prompt(
            prompt=answer,
            role="assistant"
        )
        
        # Return as a list (PyRIT 0.11.0 expects list[Message])
        return [response_message]
    
    def is_json_response_supported(self) -> bool:
        """
        Indicates whether the target supports JSON response format.
        Required abstract method from PromptTarget.
        
        Returns:
            False - The agent returns conversational text or structured responses, not raw JSON
        """
        return False

print("✓ AgentWrapper loaded successfully for PyRIT 0.11.0")