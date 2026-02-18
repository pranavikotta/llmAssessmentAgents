import asyncio
from dotenv import load_dotenv
import os

load_dotenv()

# 1. SETUP MEMORY FIRST - FIXED FOR PYRIT 0.11.0
from pyrit.memory import CentralMemory, SQLiteMemory 
CentralMemory.set_memory_instance(SQLiteMemory(db_path=":memory:"))

# 2. NOW IMPORT CUSTOM CLASSES
# Don't import get_nemo_app here - it's handled internally by agent_wrapper
from agent_wrapper import AgentTarget

async def main():
    # Import the LangGrpyaph workflow (not NeMo)
    from graphs import workflow
    
    # Create the LangGraph chatbot workflow
    chatbot_workflow = workflow()
    
    # Import and run the audit
    from run_orchestrator import run_audit
    await run_audit(chatbot_app=chatbot_workflow)

if __name__ == "__main__":
    asyncio.run(main())