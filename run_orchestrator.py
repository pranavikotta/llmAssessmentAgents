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

async def run_audit(chatbot_app):
    # Initialize PyRIT with SQLite
    await initialize_pyrit_async(memory_db_type=SQLITE)
    
    # Initialize targets - using gemini-2.0-flash
    attacker_llm = scorer.GeminiNativeTarget(model_name="gemini-2.0-flash")
    target = AgentTarget(agent=chatbot_app)
    
    with open("customer_personas.yaml", "r", encoding="utf-8") as f:
        persona_config = yaml.safe_load(f)["personas"]

    persona_to_scorer = {
        "p1": scorer.p1_leak_scorer, "p2": scorer.p2_location_scorer,
        "p3": scorer.p3_competitor_scorer, "p4": scorer.p4_scope_scorer,
        "p5": scorer.p5_physics_scorer, "p6": scorer.p6_format_scorer,
        "p7": scorer.p7_logic_scorer, "p8": scorer.p8_security_scorer,
    }

    completed_personas = []
    failed_personas = []
    
    try:
        total = len(persona_config)
        for idx, (persona_id, data) in enumerate(persona_config.items(), 1):
            print(f"\n{'='*60}")
            print(f"PERSONA {idx}/{total}: {persona_id.upper()}")
            print(f"Test: {data.get('test_type', 'Unknown')}")
            print(f"Objective: {data['objective']}")
            print('='*60 + "\n")
            
            try:
                scoring_config = AttackScoringConfig(
                    objective_scorer=persona_to_scorer.get(persona_id)
                )

                adversarial_config = AttackAdversarialConfig(
                    target=attacker_llm, 
                    system_prompt_path=RTASystemPromptPaths.TEXT_GENERATION.value 
                )

                attack = RedTeamingAttack(
                    objective_target=target,
                    attack_adversarial_config=adversarial_config,
                    attack_scoring_config=scoring_config,
                    max_turns=3
                )

                await attack.execute_async(objective=data["objective"])
                
                completed_personas.append(persona_id)
                print(f"\n{persona_id.upper()} - COMPLETE")
                
            except Exception as e:
                failed_personas.append((persona_id, str(e)))
                print(f"\n{persona_id.upper()} - FAILED: {e}")
                
                # Check if it's a rate limit error
                if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                    print("\nRate limit hit. Results so far are saved.")
                    print("   Wait for quota reset and run again to continue.")
                    break  # Stop testing on rate limit
                else:
                    print(f"   Continuing to next persona...")
                    continue  # Try next persona

        # Final summary
        print("\n" + "="*60)
        print("AUDIT SUMMARY")
        print("="*60)
        print(f"Completed: {len(completed_personas)}/{total}")
        if completed_personas:
            print(f"   {', '.join(completed_personas)}")
        
        if failed_personas:
            print(f"\nFailed: {len(failed_personas)}/{total}")
            for pid, error in failed_personas:
                print(f"   {pid}: {error[:80]}...")
        
        print("="*60)

    finally:
        # Graceful cleanup
        try:
            await attacker_llm.client.aio.aclose()
            await scorer.judge_llm.client.aio.aclose()
        except:
            pass

if __name__ == "__main__":
    from graphs import workflow
    app = workflow()
    asyncio.run(run_audit(app))