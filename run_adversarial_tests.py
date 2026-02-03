from pyrit.setup import initialize_pyrit_async, SQLITE
from pyrit.memory import CentralMemory

async def setup_pipeline():
    #initialize PyRIT Memory (creates pyrit_memory.db in current directory)
    await initialize_pyrit_async(memory_db_type=SQLITE)
    
    #access the memory instance
    memory = CentralMemory.get_memory_instance()
    
    print("Memory initialized. All battle logs will be saved to local SQLite.")