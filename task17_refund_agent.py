
import csv
import json
import sqlite3
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

import mock_llm
from task16_policy_rag import retrieve


BASE_DIR = Path(__file__).resolve().parent
DB_PATH = BASE_DIR / "data" / "instamart.db"
RAW_DIR = BASE_DIR / "data" / "raw"
OUTPUT_DIR = BASE_DIR / "outputs"
STATE_DIR = BASE_DIR / "state"

AGENT_SYSTEM = """
AGENT
You are an Instamart refund support agent.
Use the available tools to look up the order, search the policy,
calculate the refund and issue or escalate it.
Only completed orders can receive refunds.
Orders delivered in more than 45 minutes qualify for a full refund.
Orders delivered in 45 minutes or less receive no refund.
Refunds above INR 1000 must be escalated to a human.
Follow tool results and do not invent order information.
"""


# TASK 5: TOOL SPECIFICATIONS

TOOLS_SPEC = [
    {
        "type": "function",
        "function": {
            "name": "get_order",
            "description": "Look up an order in the SQLite database.",
            "parameters": {
                "type": "object",
                "properties": {
                    "order_id": {"type": "string"}
                },
                "required": ["order_id"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "search_policy",
            "description": "Search the policy documents using retrieval.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"}
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "calculate_refund",
            "description": "Calculate the eligible refund amount.",
            "parameters": {
                "type": "object",
                "properties": {
                    "order_amount": {"type": "number"},
                    "delivery_minutes": {"type": "number"}
                },
                "required": ["order_amount", "delivery_minutes"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "issue_refund",
            "description": "Record an automatically approved refund.",
            "parameters": {
                "type": "object",
                "properties": {
                    "order_id": {"type": "string"},
                    "refund_inr": {"type": "number"}
                },
                "required": ["order_id", "refund_inr"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "escalate_to_human",
            "description": "Record a refund requiring human approval.",
            "parameters": {
                "type": "object",
                "properties": {
                    "order_id": {"type": "string"},
                    "refund_inr": {"type": "number"}
                },
                "required": ["order_id", "refund_inr"]
            }
        }
    }
]


# TOOL 1: DATABASE ORDER LOOKUP

def get_order(order_id):
    with sqlite3.connect(DB_PATH) as connection:
        connection.row_factory = sqlite3.Row

        row = connection.execute(
            """
            SELECT order_id, status, order_amount, delivery_minutes
            FROM orders
            WHERE order_id = ?
            """,
            (order_id,)
        ).fetchone()

    if row is None:
        return {"error": "order not found"}

    delivery_value = row["delivery_minutes"]

    return {
        "order_id": row["order_id"],
        "status": row["status"],
        "order_amount": float(row["order_amount"]),
        "delivery_minutes": (
            int(delivery_value)
            if delivery_value is not None
            else None
        )
    }


# TOOL 2: POLICY SEARCH

def search_policy(query):
    return {"results": retrieve(query, k=3)}


# TOOL 3: REFUND CALCULATION

def calculate_refund(order_amount, delivery_minutes):
    # A NULL delivery time means the database has no confirmed delivery
    # duration. Do not approve a refund when eligibility cannot be verified.
    if delivery_minutes is None:
        refund = 0.0
    elif float(delivery_minutes) <= 45:
        refund = 0.0
    else:
        refund = round(float(order_amount), 2)

    return {"refund_inr": refund}


# REFUND LEDGER

def write_ledger(order_id, refund_inr, tool_name):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "order_id": order_id,
        "refund_inr": round(float(refund_inr), 2),
        "tool_name": tool_name
    }

    with (OUTPUT_DIR / "refund_ledger.jsonl").open(
        "a", encoding="utf-8"
    ) as file:
        file.write(json.dumps(entry) + "\n")

    return {"refund_inr": round(float(refund_inr), 2)}


# TOOL 4: ISSUE REFUND

def issue_refund(order_id, refund_inr):
    return write_ledger(
        order_id, refund_inr, "issue_refund"
    )


# TOOL 5: HUMAN ESCALATION

def escalate_to_human(order_id, refund_inr):
    return write_ledger(
        order_id, refund_inr, "escalate_to_human"
    )


TOOL_MAP = {
    "get_order": get_order,
    "search_policy": search_policy,
    "calculate_refund": calculate_refund,
    "issue_refund": issue_refund,
    "escalate_to_human": escalate_to_human
}


# TASK 6 AND 8: CHECKPOINT SAVE

def save_checkpoint(
    run_id,
    scenario_id,
    messages,
    tokens_used,
    step_number,
    executed_tools,
    steps,
    llm_calls
):
    STATE_DIR.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "run_id": run_id,
        "scenario_id": scenario_id,
        "messages": messages,
        "tokens_used": tokens_used,
        "step_number": step_number,
        "executed_tools": executed_tools,
        "steps": steps,
        "llm_calls": llm_calls
    }

    path = STATE_DIR / f"{run_id}.json"

    path.write_text(
        json.dumps(checkpoint, indent=2),
        encoding="utf-8"
    )


def load_checkpoint(run_id):
    path = STATE_DIR / f"{run_id}.json"

    if not path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {path}"
        )

    return json.loads(
        path.read_text(encoding="utf-8")
    )


# TASK 7: SAFE FALLBACK

def fallback_result(reason, steps, llm_calls, tokens_used):
    return {
        "status": reason,
        "final_answer": (
            f"{reason}: Agent stopped safely. "
            "Please hand this request to a human support agent. "
            f"Completed tool steps: {len(steps)}; "
            f"model calls: {llm_calls}; "
            f"tokens used: {tokens_used}."
        ),
        "steps": steps,
        "llm_calls": llm_calls,
        "tokens_used": tokens_used
    }


# REACT AGENT LOOP

def continue_agent(
    run_id,
    checkpoint,
    max_iterations=6,
    token_limit=6000,
    time_limit_seconds=10,
    crash_after_tools=None
):
    scenario_id = checkpoint["scenario_id"]
    messages = checkpoint["messages"]
    tokens_used = checkpoint["tokens_used"]
    steps = checkpoint.get("steps", [])
    executed_tools = checkpoint["executed_tools"]
    llm_calls = checkpoint.get("llm_calls", 0)

    started_at = time.monotonic()

    while llm_calls < max_iterations:

        if time.monotonic() - started_at >= time_limit_seconds:
            return fallback_result(
                "FALLBACK_TIMEOUT",
                steps, llm_calls, tokens_used
            )

        response = mock_llm.chat(
            messages,
            tools=TOOLS_SPEC
        )

        llm_calls += 1
        tokens_used += response.usage.total_tokens

        if tokens_used > token_limit:
            return fallback_result(
                "FALLBACK_TOKEN_LIMIT",
                steps, llm_calls, tokens_used
            )

        assistant_message = response.choices[0].message
        tool_calls = assistant_message.tool_calls or []

        if not tool_calls:
            final_answer = assistant_message.content or ""
            status = final_answer.split(":", 1)[0].strip()

            return {
                "status": status,
                "final_answer": final_answer,
                "steps": steps,
                "llm_calls": llm_calls,
                "tokens_used": tokens_used
            }

        messages.append({
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": call.id,
                    "name": call.name,
                    "arguments": call.arguments
                }
                for call in tool_calls
            ]
        })

        for call in tool_calls:
            tool_name = call.name
            arguments = call.arguments

            if isinstance(arguments, str):
                arguments = json.loads(arguments)

            if tool_name not in TOOL_MAP:
                observation = {"error": "unknown tool"}
            else:
                observation = TOOL_MAP[tool_name](**arguments)

            messages.append({
                "role": "tool",
                "tool_call_id": call.id,
                "name": tool_name,
                "content": json.dumps(observation)
            })

            step = {
                "tool": tool_name,
                "arguments": arguments,
                "observation": observation
            }

            steps.append(step)
            executed_tools.append(tool_name)

            # Save the full progress after every tool observation.
            save_checkpoint(
                run_id,
                scenario_id,
                messages,
                tokens_used,
                len(steps),
                executed_tools,
                steps,
                llm_calls
            )

            if (
                crash_after_tools is not None
                and len(steps) == crash_after_tools
            ):
                return {
                    "status": "CRASH_SIMULATED",
                    "final_answer": "Simulated crash after checkpoint save.",
                    "steps": steps,
                    "llm_calls": llm_calls,
                    "tokens_used": tokens_used,
                    "run_id": run_id
                }

    return fallback_result(
        "FALLBACK_MAX_ITERATIONS",
        steps, llm_calls, tokens_used
    )


# START A NEW AGENT RUN

def run_agent(
    user_message,
    scenario_id,
    max_iterations=6,
    token_limit=6000,
    time_limit_seconds=10,
    crash_after_tools=None
):
    run_id = f"{scenario_id}_{uuid.uuid4().hex[:10]}"

    messages = [
        {"role": "system", "content": AGENT_SYSTEM},
        {"role": "user", "content": user_message}
    ]

    checkpoint = {
        "run_id": run_id,
        "scenario_id": scenario_id,
        "messages": messages,
        "tokens_used": 0,
        "step_number": 0,
        "executed_tools": [],
        "steps": [],
        "llm_calls": 0
    }

    return continue_agent(
        run_id,
        checkpoint,
        max_iterations,
        token_limit,
        time_limit_seconds,
        crash_after_tools
    )


# RESUME AN EXISTING RUN

def resume_agent(
    run_id,
    max_iterations=6,
    token_limit=6000,
    time_limit_seconds=10
):
    checkpoint = load_checkpoint(run_id)

    return continue_agent(
        run_id,
        checkpoint,
        max_iterations,
        token_limit,
        time_limit_seconds
    )


# JSONL OUTPUT HELPERS

def append_jsonl(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(data) + "\n")


def save_agent_run(scenario_id, result):
    entry = {
        "scenario_id": scenario_id,
        "status": result["status"],
        "final_answer": result["final_answer"],
        "llm_calls": result["llm_calls"],
        "tokens_used": result["tokens_used"],
        "steps": result["steps"]
    }

    append_jsonl(
        OUTPUT_DIR / "agent_runs.jsonl",
        entry
    )


# LOAD SCENARIOS

def read_scenarios():
    path = RAW_DIR / "agent_scenarios.csv"

    with path.open(
        "r", encoding="utf-8-sig", newline=""
    ) as file:
        return list(csv.DictReader(file))


# TASK 6: RUN SEVEN SCENARIOS

def run_all_scenarios():
    scenarios = read_scenarios()

    print("\nTASK 6 - SEVEN AGENT SCENARIOS")
    print("=" * 55)

    results = []

    for scenario in scenarios:
        scenario_id = scenario["scenario_id"]

        mock_llm.reset()

        result = run_agent(
            scenario["message"],
            scenario_id,
            max_iterations=6,
            token_limit=6000,
            time_limit_seconds=10
        )

        save_agent_run(scenario_id, result)
        results.append(result)

        print(
            f"{scenario_id}: {result['status']} | "
            f"Calls: {result['llm_calls']} | "
            f"Tokens: {result['tokens_used']}"
        )

        print("Answer:", result["final_answer"])

    return scenarios, results


# TASK 7: SAFETY DEMONSTRATIONS

def run_safety_demos(scenarios):
    print("\nTASK 7 - SAFE STOP DEMONSTRATIONS")
    print("=" * 55)

    s1 = next(
        row for row in scenarios
        if row["scenario_id"] == "S1"
    )

    mock_llm.reset()

    token_demo = run_agent(
        s1["message"],
        "S1_TOKEN_DEMO",
        token_limit=200
    )

    print("Token limit demo:", token_demo["status"])
    print(token_demo["final_answer"])

    mock_llm.reset()

    timeout_demo = run_agent(
        s1["message"],
        "S1_TIMEOUT_DEMO",
        time_limit_seconds=0
    )

    print("Timeout demo:", timeout_demo["status"])
    print(timeout_demo["final_answer"])

    return token_demo, timeout_demo


# TASK 8: CRASH AND RESUME

def run_resume_demo(scenarios):
    print("\nTASK 8 - CHECKPOINT AND RESUME")
    print("=" * 55)

    s1 = next(
        row for row in scenarios
        if row["scenario_id"] == "S1"
    )

    mock_llm.reset()

    crashed = run_agent(
        s1["message"],
        "S1_RESUME_DEMO",
        crash_after_tools=2
    )

    run_id = crashed["run_id"]

    print("Simulated crash:", crashed["status"])
    print(
        "Tools before crash:",
        [step["tool"] for step in crashed["steps"]]
    )

    resumed = resume_agent(run_id)

    print("Resumed status:", resumed["status"])
    print("Final answer:", resumed["final_answer"])

    print(
        "Tools across crash and resume:",
        [step["tool"] for step in resumed["steps"]]
    )

    return resumed


# MAIN PROGRAM

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    STATE_DIR.mkdir(parents=True, exist_ok=True)

    print("PART D - REFUND REACT AGENT")
    print("=" * 55)

    scenarios, results = run_all_scenarios()

    token_demo, timeout_demo = run_safety_demos(
        scenarios
    )

    resumed = run_resume_demo(scenarios)

    print("\nFINAL SUMMARY")
    print("=" * 55)

    for scenario, result in zip(scenarios, results):
        print(
            scenario["scenario_id"],
            result["status"],
            result["final_answer"]
        )

    print("\nSafety demo statuses:")
    print("Token limit:", token_demo["status"])
    print("Timeout:", timeout_demo["status"])

    print("\nResume final status:", resumed["status"])
    print("Resume final answer:", resumed["final_answer"])

    print("\nTask 5 to Task 8 execution completed.")
    print("Agent runs:", OUTPUT_DIR / "agent_runs.jsonl")
    print("Refund ledger:", OUTPUT_DIR / "refund_ledger.jsonl")
    print("Checkpoints:", STATE_DIR)


if __name__ == "__main__":
    main()
