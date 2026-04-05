"""
dspy_comparison_test.py — Measures the impact of DSPy optimization on chatbot defense.

Runs all 8 personas against two conditions:
  - Condition A: WITH DSPy optimization (optimized_chatbot.json)
  - Condition B: WITHOUT DSPy optimization (base AIAssistant)

Results saved to: dspy_comparison_tagged.db with a 'condition' column.

FIX: Avoids importlib.reload entirely. Instead builds a fresh workflow
per condition by directly constructing the AIAssistant and passing it
into the graph, bypassing the module-level singleton.
"""

import asyncio
import yaml
import os
import sqlite3
import json
import re
from datetime import datetime
from typing import TypedDict, List, Optional
from typing_extensions import Annotated
from dotenv import load_dotenv
load_dotenv()

# Must be set before prompt_optimization is imported (directly or transitively),
# so the module-level singleton is skipped. build_workflow() manages its own
# AIAssistant instances per condition instead.
os.environ["PROMPT_OPT_INIT"] = "false"

from pyrit.memory import CentralMemory, SQLiteMemory
CentralMemory.set_memory_instance(SQLiteMemory(db_path="dspy_pyrit_log.db"))

from agent_wrapper import AgentTarget
import scorer
from pyrit.executor.attack import (
    AttackAdversarialConfig,
    AttackScoringConfig,
    RedTeamingAttack,
    RTASystemPromptPaths,
)
from pyrit.setup import initialize_pyrit_async, SQLITE

# ── CONFIG ────────────────────────────────────────────────────────────────────
ROUNDS = 5
MAX_TURNS = 3
PERSONAS_TO_TEST = ["p1", "p2", "p3", "p4", "p5", "p6", "p7", "p8"]
RESULTS_DB = "dspy_comparison_tagged.db"
# ─────────────────────────────────────────────────────────────────────────────

PERSONA_SCORERS = {
    "p1": scorer.p1_leak_scorer,
    "p2": scorer.p2_location_scorer,
    "p3": scorer.p3_competitor_scorer,
    "p4": scorer.p4_scope_scorer,
    "p5": scorer.p5_physics_scorer,
    "p6": scorer.p6_format_scorer,
    "p7": scorer.p7_logic_scorer,
    "p8": scorer.p8_security_scorer,
}

PERSONA_LABELS = {
    "p1": "adversarial_injection",
    "p2": "vague_location",
    "p3": "competitor_probing",
    "p4": "out_of_scope",
    "p5": "technical_stress",
    "p6": "format_sabotage",
    "p7": "intent_collision",
    "p8": "unauthorized_access",
}


def init_results_db():
    conn = sqlite3.connect(RESULTS_DB)
    conn.execute('''
        CREATE TABLE IF NOT EXISTS dspy_results (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            persona_id TEXT,
            test_type TEXT,
            condition TEXT,
            round_num INTEGER,
            objective TEXT,
            outcome TEXT,
            attacker_succeeded INTEGER,
            turns INTEGER,
            execution_time_ms INTEGER,
            score_value TEXT,
            score_rationale TEXT,
            timestamp TEXT
        )
    ''')
    conn.commit()
    conn.close()


def save_round_result(persona_id, condition, round_num, objective,
                      outcome, attacker_succeeded, turns, time_ms,
                      score_value, score_rationale):
    conn = sqlite3.connect(RESULTS_DB)
    conn.execute('''
        INSERT INTO dspy_results
        (persona_id, test_type, condition, round_num, objective, outcome,
         attacker_succeeded, turns, execution_time_ms, score_value, score_rationale, timestamp)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (
        persona_id,
        PERSONA_LABELS.get(persona_id, "unknown"),
        condition,
        round_num,
        objective,
        outcome,
        1 if attacker_succeeded else 0,
        turns,
        time_ms,
        score_value,
        score_rationale,
        datetime.now().isoformat()
    ))
    conn.commit()
    conn.close()


def build_workflow(use_optimization: bool):
    """
    Build a fresh chatbot_only_workflow with its own AIAssistant instance.
    Avoids importlib.reload — constructs the assistant directly per condition.
    Uses assistant(...) call syntax (not .forward()) to avoid DSPy warning.
    """
    from langgraph.graph.message import add_messages
    from langgraph.graph import StateGraph, END
    from langchain_core.messages import AIMessage
    from agent import chatbot_llm
    from prompt_optimization import AIAssistant

    # Build a fresh assistant for this condition
    if use_optimization and os.path.exists("optimized_chatbot.json"):
        assistant = AIAssistant()
        assistant.load("optimized_chatbot.json")
        print(f"  [build_workflow] loaded optimized_chatbot.json ✅")
    else:
        assistant = AIAssistant()
        print(f"  [build_workflow] using base unoptimized assistant ⬜")

    class ChatTesterState(TypedDict):
        messages: Annotated[list, add_messages]
        api_calls: Optional[List[str]]
        test_type: str
        next_agent: str
        chat_finished: bool

    def check_for_json(response):
        content = response.content.strip()
        try:
            return [json.loads(content)]
        except json.JSONDecodeError:
            match = re.search(r"\{.*\}", content, re.DOTALL)
            if match:
                try:
                    return [json.loads(match.group(0))]
                except:
                    return []
        return []

    def chatbot_node(state: ChatTesterState) -> ChatTesterState:
        messages = state["messages"]
        last_msg = messages[-1].content
        history_str = "\n".join([f"{m.type}: {m.content}" for m in messages[:-1]])

        check_query = f"Does this message contain a specific location? Answer Yes or No: {last_msg}"
        intent_result = chatbot_llm.invoke(check_query).content.lower()
        location_detected = "yes" in intent_result

        if location_detected:
            print("Mode: Recommendations (JSON)")
            catalogue_data = "Maxar-15cm|pid:644b|oid:0134\nICEYE-SAR|pid:992a|oid:0882"
            # Use __call__ not .forward() to avoid DSPy warning and arg issues
            # NOTE: kwarg must be product_catalogue to match AIAssistant.forward() and Recommender signature
            dspy_output = assistant(
                question=last_msg,
                history=history_str,
                product_catalogue=catalogue_data,
                location_detected=True
            )
            content_output = dspy_output.answer
        else:
            print("Mode: General QA (Text)")
            result = assistant(
                question=last_msg,
                history=history_str,
                location_detected=False
            )
            content_output = result.answer
            if hasattr(result, 'reasoning') and result.reasoning:
                print(f"DEBUG REASONING: {result.reasoning}")

        response = AIMessage(content=content_output)
        new_extracted = check_for_json(response)
        existing_calls = state.get("api_calls") or []

        return {
            "messages": [response],
            "next_agent": "customer",
            "chat_finished": False,
            "test_type": state["test_type"],
            "api_calls": existing_calls + new_extracted
        }

    def route_logic(state: ChatTesterState) -> str:
        if "test_complete" in state["messages"][-1].content.lower():
            return "END"
        if len(state["messages"]) >= 15:
            return "END"
        return state["next_agent"]

    wf = StateGraph(ChatTesterState)
    wf.add_node("chatbot", chatbot_node)
    wf.add_conditional_edges("chatbot", route_logic, {"customer": END, "END": END})
    wf.set_entry_point("chatbot")
    return wf.compile()


async def run_condition(persona_id, condition, objective, persona_system_prompt,
                        persona_scorer, target, rounds):
    attacker_llm = scorer.get_llm_target()
    attacker_llm.set_system_prompt(persona_system_prompt)
    successes = 0

    for i in range(rounds):
        print(f"  Round {i+1}/{rounds}...", end=" ", flush=True)
        try:
            scoring_config = AttackScoringConfig(objective_scorer=persona_scorer)
            adversarial_config = AttackAdversarialConfig(
                target=attacker_llm,
                system_prompt_path=RTASystemPromptPaths.TEXT_GENERATION.value,
            )
            attack = RedTeamingAttack(
                objective_target=target,
                attack_adversarial_config=adversarial_config,
                attack_scoring_config=scoring_config,
                max_turns=MAX_TURNS,
            )

            start = datetime.now()
            result = await attack.execute_async(objective=objective)
            elapsed_ms = int((datetime.now() - start).total_seconds() * 1000)

            # Safely convert outcome enum to string
            outcome = getattr(result, 'outcome', None)
            if outcome is not None:
                outcome = str(outcome.value) if hasattr(outcome, 'value') else str(outcome)
            turns = getattr(result, 'executed_turns', MAX_TURNS)

            # Default from outcome string
            attacker_succeeded = (outcome == "success")
            score_value = "true" if attacker_succeeded else "false"
            score_rationale = ""

            # Try richer score from PyRIT memory
            try:
                mem = CentralMemory.get_memory_instance()
                last_response_id = getattr(result, 'last_response_id', None)
                if last_response_id:
                    scores = mem.get_scores_by_prompt_ids(
                        prompt_request_response_ids=[last_response_id]
                    )
                    if scores:
                        score_value = str(scores[0].score_value)
                        score_rationale = str(scores[0].score_rationale or "")[:300]
                        attacker_succeeded = score_value.lower() == "true"
            except Exception:
                pass

            if attacker_succeeded:
                successes += 1
                print("ATTACKER SUCCEEDED")
            else:
                print("AGENT HELD")

            save_round_result(
                persona_id=persona_id, condition=condition, round_num=i + 1,
                objective=objective,
                outcome=outcome or ("success" if attacker_succeeded else "failure"),
                attacker_succeeded=attacker_succeeded, turns=turns,
                time_ms=elapsed_ms, score_value=score_value,
                score_rationale=score_rationale
            )

        except Exception as e:
            err = str(e)
            print(f"ERROR — {err[:80]}")
            save_round_result(
                persona_id=persona_id, condition=condition, round_num=i + 1,
                objective=objective, outcome="error", attacker_succeeded=False,
                turns=0, time_ms=0, score_value="error", score_rationale=err[:300]
            )
            if "429" in err or "RESOURCE_EXHAUSTED" in err or "rate_limit" in err.lower():
                print("\nRate limit hit. Stopping early.")
                return successes, i + 1

    try:
        if hasattr(attacker_llm, 'client'):
            if hasattr(attacker_llm.client, 'aio'):
                await attacker_llm.client.aio.aclose()
            else:
                await attacker_llm.client.close()
    except:
        pass

    return successes, rounds


async def main():
    with open("customer_personas.yaml", "r", encoding="utf-8") as f:
        persona_config = yaml.safe_load(f)["personas"]

    init_results_db()

    print("\n" + "=" * 60)
    print("DSPy Optimization Comparison Experiment")
    print(f"Personas: ALL 8 | Rounds per condition: {ROUNDS}")
    print(f"Provider: {scorer.LLM_PROVIDER.upper()}")
    print(f"Results DB: {RESULTS_DB}")
    print("=" * 60 + "\n")

    await initialize_pyrit_async(memory_db_type=SQLITE)

    results = {}

    for persona_id in PERSONAS_TO_TEST:
        data = persona_config[persona_id]
        objective = data["objective"]
        persona_system_prompt = data["system_prompt"]

        print(f"\n{'=' * 60}")
        print(f"PERSONA: {persona_id.upper()} — {data['test_type']}")
        print(f"Objective: {objective}")
        print("=" * 60)

        results[persona_id] = {}

        for condition, use_opt in [("optimized", True), ("base", False)]:
            label = "WITH DSPy optimization" if use_opt else "WITHOUT DSPy optimization"
            print(f"\n  Condition: {label}")
            print(f"  {'-' * 40}")

            chatbot_app = build_workflow(use_optimization=use_opt)
            target = AgentTarget(agent=chatbot_app, use_nemo=False)

            successes, rounds_run = await run_condition(
                persona_id=persona_id, condition=condition, objective=objective,
                persona_system_prompt=persona_system_prompt,
                persona_scorer=PERSONA_SCORERS[persona_id],
                target=target, rounds=ROUNDS,
            )
            results[persona_id][condition] = (successes, rounds_run)
            rate = successes / rounds_run if rounds_run > 0 else 0
            print(f"\n  {label} — Attacker success rate: {successes}/{rounds_run} ({rate:.0%})")

    # ── FINAL SUMMARY ─────────────────────────────────────────────────────────
    print("\n\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    print(f"{'Persona':<10} {'Base (no DSPy)':>16} {'Optimized':>12} {'Improvement':>14}")
    print("-" * 56)

    for persona_id in PERSONAS_TO_TEST:
        base_s, base_r = results[persona_id].get("base", (0, 0))
        opt_s, opt_r = results[persona_id].get("optimized", (0, 0))
        base_rate = base_s / base_r if base_r > 0 else 0
        opt_rate = opt_s / opt_r if opt_r > 0 else 0
        improvement = base_rate - opt_rate
        symbol = "✅" if improvement > 0 else ("➡️" if improvement == 0 else "❌")
        print(
            f"{symbol} {persona_id.upper():<8}"
            f"{base_rate:>15.0%}"
            f"{opt_rate:>12.0%}"
            f"{improvement:>+14.0%}"
        )

    print("=" * 60)
    print(f"\nTagged results saved to: {RESULTS_DB}")
    print("Run: python analyze_results.py dspy_comparison_tagged.db")
    print("Positive improvement = DSPy optimization reduced attacker success rate\n")

    try:
        if hasattr(scorer.judge_llm.client, 'aio'):
            await scorer.judge_llm.client.aio.aclose()
        else:
            await scorer.judge_llm.client.close()
    except:
        pass


if __name__ == "__main__":
    asyncio.run(main())