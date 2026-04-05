import asyncio
import yaml
import os
from pyrit.executor.attack import (
    AttackAdversarialConfig,
    AttackScoringConfig,
    RedTeamingAttack,
    RTASystemPromptPaths,
)
from pyrit.setup import initialize_pyrit_async, SQLITE
from agent_wrapper import AgentTarget
import scorer

async def run_audit(chatbot_app):
    await initialize_pyrit_async(memory_db_type=SQLITE)

    # Provider-aware: uses Grok or Gemini based on LLM_PROVIDER env var
    attacker_llm = scorer.get_llm_target()
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
                    system_prompt_path=RTASystemPromptPaths.TEXT_GENERATION.value,
                )
                attack = RedTeamingAttack(
                    objective_target=target,
                    attack_adversarial_config=adversarial_config,
                    attack_scoring_config=scoring_config,
                    max_turns=3,
                )
                await attack.execute_async(objective=data["objective"])
                completed_personas.append(persona_id)
                print(f"\n{persona_id.upper()} - COMPLETE")

            except Exception as e:
                failed_personas.append((persona_id, str(e)))
                print(f"\n{persona_id.upper()} - FAILED: {e}")
                if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e) or "rate_limit" in str(e).lower():
                    print("\nRate limit hit. Results so far are saved.")
                    break
                else:
                    continue

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
        try:
            if hasattr(attacker_llm, 'client'):
                if hasattr(attacker_llm.client, 'aio'):
                    await attacker_llm.client.aio.aclose()
                else:
                    await attacker_llm.client.close()
            if hasattr(scorer.judge_llm, 'client'):
                if hasattr(scorer.judge_llm.client, 'aio'):
                    await scorer.judge_llm.client.aio.aclose()
                else:
                    await scorer.judge_llm.client.close()
        except:
            pass

if __name__ == "__main__":
    from graphs import workflow
    app = workflow()
    asyncio.run(run_audit(app))