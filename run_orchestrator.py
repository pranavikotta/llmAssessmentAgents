import asyncio
import yaml
import os
from pyrit.executor.attack import (
    AttackAdversarialConfig,
    AttackScoringConfig,
    RedTeamingAttack,
    RTASystemPromptPaths
)
from pyrit.setup import initialize_pyrit_async, SQLITE
from agent_wrapper import AgentTarget
import scorer

with open("customer_personas.yaml", "r", encoding="utf-8") as f:
    persona_config = yaml.safe_load(f)["personas"]

async def run_audit(chatbot_app):
    #initialize pyrit memory
    await initialize_pyrit_async(memory_db_type=SQLITE)
    
    target = AgentTarget(agent=chatbot_app)
    
    #map persona IDs to the pyrit scorers
    persona_to_scorer = {
        "p1": scorer.p1_leak_scorer,
        "p2": scorer.p2_location_scorer,
        "p3": scorer.p3_competitor_scorer,
        "p4": scorer.p4_scope_scorer,
        "p5": scorer.p5_physics_scorer,
        "p6": scorer.p6_format_scorer,
        "p7": scorer.p7_logic_scorer,
        "p8": scorer.p8_security_scorer,
    }

    for persona_id, data in persona_config.items():
        print(f"\nLaunching Attack: {persona_id} ({data.get('test_type', 'Unnamed')})")
        
        #setup scoring configuration
        scoring_config = AttackScoringConfig(
            objective_scorer=persona_to_scorer.get(persona_id, scorer.p4_scope_scorer)
        )

        #setup adversarial LLM, uses Gemini Judge and standard PyRIT attack prompts
        adversarial_config = AttackAdversarialConfig(
            target=scorer.judge_llm, 
            system_prompt_path=RTASystemPromptPaths.TEXT_GENERATION.value 
        )

        #create the attack executor
        attack = RedTeamingAttack(
            objective_target=target,
            attack_adversarial_config=adversarial_config,
            attack_scoring_config=scoring_config,
            max_turns=6
        )

        #execute the attack using your persona's objective
        print(f"Objective: {data['objective']}")
        result = await attack.execute_async(objective=data["objective"])
        
        #log summary to terminal
        status = "BREACH DETECTED" if result.achieved_objective else "SECURE"
        print(f"RESULT for {persona_id}: {status}")

if __name__ == "__main__":
    #this block runs the script independently for debugging
    from graphs import workflow
    app = workflow()
    asyncio.run(run_audit(app))