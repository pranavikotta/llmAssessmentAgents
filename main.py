# main.py
import asyncio
from graphs import workflow
from run_orchestrator import run_audit

async def main():
    print("starting agent suite...")
    app = workflow()
    
    #run baseline customer agent (optional)
    # response = app.invoke({"messages": [HumanMessage(content="Hello")], "test_type": "p8"})
    
    #run PyRIT audit
    await run_audit(app)

if __name__ == "__main__":
    asyncio.run(main())