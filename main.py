import asyncio
from dotenv import load_dotenv
import os

load_dotenv()

# 1. SETUP MEMORY FIRST - SAVE TO FILE FOR PERSISTENCE
from pyrit.memory import CentralMemory, SQLiteMemory 
CentralMemory.set_memory_instance(SQLiteMemory(db_path="pyrit_results.db"))

# 2. NOW IMPORT CUSTOM CLASSES
from agent_wrapper import AgentTarget

async def main():
    print("\n" + "="*60)
    print("PyRIT Security Audit - Starting")
    print("="*60)
    print("Results will be saved to: pyrit_results.db")
    print("="*60 + "\n")
    
    try:
        # Import the LangGraph workflow (not NeMo)
        from graphs import workflow
        
        # Create the LangGraph chatbot workflow
        chatbot_workflow = workflow()
        
        # Import and run the audit
        from run_orchestrator import run_audit
        await run_audit(chatbot_app=chatbot_workflow)
        
        print("\n" + "="*60)
        print("Audit Complete!")
        print("="*60)
        print("Results saved to: pyrit_results.db")
        print("Run 'python inspect_results.py' to view results")
        print("="*60 + "\n")
        
    except Exception as e:
        print(f"\nError during audit: {e}")
        import traceback
        traceback.print_exc()
        print("\nPartial results may be saved in: pyrit_results.db")

if __name__ == "__main__":
    asyncio.run(main())