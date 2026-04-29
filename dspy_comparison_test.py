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

from pyrit.memory import CentralMemory, SQLiteMemory
CentralMemory.set_memory_instance(SQLiteMemory(db_path="dspy_pyrit_log.db"))

from agent_wrapper import AgentTarget, _json_to_plaintext
import scorer
from logger import ResultLogger, RoundRecord, TurnLogger
from pyrit.executor.attack import (
    AttackAdversarialConfig,
    AttackScoringConfig,
    RedTeamingAttack,
    RTASystemPromptPaths,
)
from pyrit.setup import initialize_pyrit_async, SQLITE

# ── CONFIG ────────────────────────────────────────────────────────────────────
ROUNDS = 15
MAX_TURNS = 3
PERSONAS_TO_TEST = ["p1", "p9"]
RESULTS_DB = "dspy_comparison_tagged.db"
CSV_LOG    = "results_log.csv"
EXPERIMENT_TAG = "p1_p8_v1"   # change per experiment
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
    "p9": scorer.p9_roleplay_scorer,
    "p10": scorer.p10_escalation_scorer,
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
    "p9": "roleplay_escape",
    "p10": "gradual_escalation",
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


def build_workflow(use_optimization: bool, force_recommender: bool = False):
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
        turn_log: Optional[List[dict]]   # per-turn mode + reasoning records

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

        reasoning_text = ""

        if force_recommender:
            location_detected = True
            print("Mode: FORCED Recommendations (P2/P5 bypass)")
        else:
            check_query = f"Does this message contain a specific location? Answer Yes or No: {last_msg}"
            intent_result = chatbot_llm.invoke(check_query).content.lower()
            location_detected = "yes" in intent_result

        if location_detected:
            mode = "REC"
            print("Mode: Recommendations (JSON)")
            catalogue_data = "Maxar-15cm|pid:644b|oid:0134\nICEYE-SAR|pid:992a|oid:0882"
            dspy_output = assistant(
                question=last_msg,
                history=history_str,
                product_catalogue=catalogue_data,
                location_detected=True
            )
            content_output = dspy_output.answer
        else:
            mode = "QA"
            print("Mode: General QA (Text)")
            result = assistant(
                question=last_msg,
                history=history_str,
                location_detected=False
            )
            content_output = result.answer
            if hasattr(result, "reasoning") and result.reasoning:
                reasoning_text = result.reasoning
                print(f"DEBUG REASONING: {reasoning_text}")

        # Record this turn for the logger
        existing_log = list(state.get("turn_log") or [])
        turn_num = len(existing_log) + 1
        TurnLogger.record(existing_log, turn_num, mode, reasoning_text)

        # CRITICAL FIX for P2/P5: convert recommender JSON to plaintext BEFORE
        # storing in LangGraph state. Without this, raw JSON leaks into conversation
        # history, causing the attacker LLM to mimic JSON format in its replies,
        # which crashes PyRIT's JSONAdapter with AdapterParseError.
        display_content = _json_to_plaintext(content_output) if mode == "REC" else content_output
        response = AIMessage(content=display_content)
        new_extracted = check_for_json(response)
        existing_calls = state.get("api_calls") or []

        return {
            "messages": [response],
            "next_agent": "customer",
            "chat_finished": False,
            "test_type": state["test_type"],
            "api_calls": existing_calls + new_extracted,
            "turn_log": existing_log,
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
                        persona_scorer, target, rounds, result_logger,
                        expected_mode="QA"):
    """
    expected_mode: "QA" for personas targeting general_qa,
                   "REC" for personas targeting recommendations.
    Used to calculate mode_confusion (unexpected REC triggers).
    """
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

            outcome = getattr(result, "outcome", None)
            if outcome is not None:
                outcome = str(outcome.value) if hasattr(outcome, "value") else str(outcome)
            turns = getattr(result, "executed_turns", MAX_TURNS)

            attacker_succeeded = (outcome == "success")
            score_value = "true" if attacker_succeeded else "false"
            score_float = None
            score_rationale = ""

            try:
                mem = CentralMemory.get_memory_instance()
                last_response_id = getattr(result, "last_response_id", None)
                if last_response_id:
                    scores = mem.get_scores_by_prompt_ids(
                        prompt_request_response_ids=[last_response_id]
                    )
                    if scores:
                        score_value = str(scores[0].score_value)
                        score_rationale = str(scores[0].score_rationale or "")[:300]
                        attacker_succeeded = score_value.lower() == "true"
                        # Try to parse a float score if available
                        try:
                            score_float = float(scores[0].score_value)
                        except (ValueError, TypeError):
                            score_float = 1.0 if attacker_succeeded else 0.0
            except Exception:
                pass

            if attacker_succeeded:
                successes += 1
                print("ATTACKER SUCCEEDED")
            else:
                print("AGENT HELD")

            # Collect turn_log from the AgentTarget (populated by chatbot_node)
            turn_log = getattr(target, "last_turn_log", [])

            # Build and save RoundRecord via logger
            from logger import RoundRecord
            record = RoundRecord(
                run_id=result_logger.run_id,
                experiment_tag=result_logger.experiment_tag,
                persona_id=persona_id,
                test_type=PERSONA_LABELS.get(persona_id, "unknown"),
                condition=condition,
                round_num=i + 1,
                objective=objective,
                outcome=outcome or ("success" if attacker_succeeded else "failure"),
                attacker_succeeded=1 if attacker_succeeded else 0,
                turns_used=turns or 0,
                execution_time_ms=elapsed_ms,
                score_value=score_value,
                score_float=score_float,
                score_rationale=score_rationale,
                turns=turn_log,
            )
            record.finalize(expected_mode=expected_mode)
            result_logger.log(record)

            # Also write to legacy table for backwards compat with analyze_db_results.py
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

            # Log error round too so it shows in CSV
            from logger import RoundRecord
            record = RoundRecord(
                run_id=result_logger.run_id,
                experiment_tag=result_logger.experiment_tag,
                persona_id=persona_id,
                test_type=PERSONA_LABELS.get(persona_id, "unknown"),
                condition=condition,
                round_num=i + 1,
                objective=objective,
                outcome="error",
                attacker_succeeded=0,
                turns_used=0,
                execution_time_ms=0,
                score_value="error",
                score_float=None,
                score_rationale=err[:300],
                turns=[],
            )
            record.finalize(expected_mode=expected_mode)
            result_logger.log(record)

            save_round_result(
                persona_id=persona_id, condition=condition, round_num=i + 1,
                objective=objective, outcome="error", attacker_succeeded=False,
                turns=0, time_ms=0, score_value="error", score_rationale=err[:300]
            )
            if "429" in err or "RESOURCE_EXHAUSTED" in err or "rate_limit" in err.lower():
                print("\nRate limit hit. Stopping early.")
                return successes, i + 1

    try:
        if hasattr(attacker_llm, "client"):
            if hasattr(attacker_llm.client, "aio"):
                await attacker_llm.client.aio.aclose()
            else:
                await attacker_llm.client.close()
    except Exception:
        pass

    return successes, rounds

# Maps persona_id to expected routing mode for mode_confusion calculation
PERSONA_EXPECTED_MODE = {
    "p1": "QA",   # adversarial injection — should stay in QA
    "p2": "QA",   # vague location — should stay in QA (never give valid location)
    "p3": "REC",  # competitor probing — targets recommender
    "p4": "QA",   # out of scope — should stay in QA
    "p5": "REC",  # technical stress — targets recommender
    "p6": "REC",  # format sabotage — targets recommender
    "p7": "QA",   # intent collision — should stay in QA
    "p8": "QA",   # unauthorized access — should stay in QA
    "p9": "QA",   # roleplay escape - should stay in QA
    "p10": "QA",  # gradual escalation - should stay in QA
}

async def main():
    with open("customer_personas.yaml", "r", encoding="utf-8") as f:
        persona_config = yaml.safe_load(f)["personas"]

    init_results_db()

    # Instantiate logger once for the whole run
    result_logger = ResultLogger(
        db_path=RESULTS_DB,
        csv_path=CSV_LOG,
        experiment_tag=EXPERIMENT_TAG,
    )

    print("\n" + "=" * 60)
    print("DSPy Optimization Comparison Experiment")
    print(f"Personas: ALL 8 | Rounds per condition: {ROUNDS}")
    print(f"Provider: {scorer.LLM_PROVIDER.upper()}")
    print(f"Results DB: {RESULTS_DB}")
    print(f"CSV log:    {CSV_LOG}")
    print(f"Experiment: {EXPERIMENT_TAG}  run_id: {result_logger.run_id}")
    print("=" * 60 + "\n")

    await initialize_pyrit_async(memory_db_type=SQLITE)

    results = {}

    for persona_id in PERSONAS_TO_TEST:
        data = persona_config[persona_id]
        objective = data["objective"]
        persona_system_prompt = data["system_prompt"]
        expected_mode = PERSONA_EXPECTED_MODE.get(persona_id, "QA")

        print(f"\n{'=' * 60}")
        print(f"PERSONA: {persona_id.upper()} — {data['test_type']}")
        print(f"Objective: {objective}")
        print("=" * 60)

        results[persona_id] = {}

        for condition, use_opt in [("optimized", True), ("base", False)]:
            label = "WITH DSPy optimization" if use_opt else "WITHOUT DSPy optimization"
            print(f"\n  Condition: {label}")
            print(f"  {'-' * 40}")

            chatbot_app = build_workflow(use_optimization=use_opt, force_recommender=(persona_id in ["p2", "p5"]))
            target = AgentTarget(agent=chatbot_app, use_nemo=False)

            successes, rounds_run = await run_condition(
                persona_id=persona_id, condition=condition, objective=objective,
                persona_system_prompt=persona_system_prompt,
                persona_scorer=PERSONA_SCORERS[persona_id],
                target=target, rounds=ROUNDS,
                result_logger=result_logger,
                expected_mode=expected_mode,
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
    print(f"CSV log saved to: {CSV_LOG}")
    print("Run: python analyze_db_results.py dspy_comparison_tagged.db")
    print("Positive improvement = DSPy optimization reduced attacker success rate\n")

    # Print run summary from logger
    result_logger.summary()

    try:
        if hasattr(scorer.judge_llm.client, "aio"):
            await scorer.judge_llm.client.aio.aclose()
        else:
            await scorer.judge_llm.client.close()
    except Exception:
        pass

if __name__ == "__main__":
    asyncio.run(main())