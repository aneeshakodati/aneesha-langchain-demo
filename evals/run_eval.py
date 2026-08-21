"""Run the evaluation suite against LangSmith.

    python evals/run_eval.py                          # full suite
    python evals/run_eval.py --kind refund            # one slice
    python evals/run_eval.py --model anthropic:claude-haiku-4-5-20251001
    python evals/run_eval.py --local                  # no LangSmith, print a table

The `--model` flag is the point of the whole exercise. Run the suite on Sonnet,
run it on Haiku, and the experiment comparison view answers "can we afford the
cheap model?" with a number for policy adherence instead of a vibe.

`--local` runs the identical evaluators without uploading, so the suite is usable
in CI or offline. The evaluators don't know or care which mode they're in.
"""

from __future__ import annotations

import argparse
import os
import uuid
from collections import defaultdict
from typing import Any

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.store.memory import InMemoryStore

import chinook_support  # noqa: F401  (loads .env)
from chinook_support.context import SupportContext
from chinook_support.db import query, query_one, write_conn
from chinook_support.graph import build_graph

from evals.dataset import DATASET_NAME, build_examples
from evals.evaluators import ALL_EVALUATORS

BOLD, DIM, GREEN, RED, YELLOW, RESET = (
    "\033[1m", "\033[2m", "\033[32m", "\033[31m", "\033[33m", "\033[0m",
)


def _reset_account(customer_id: int) -> None:
    """Undo side effects from earlier examples so each case starts clean.

    Without this, example 3 refunding order 414 makes example 9's verdict
    `already_refunded` and the suite grades correct behavior as a failure. Eval
    isolation is unglamorous and it is the difference between a suite you trust
    and one you learn to ignore.
    """
    with write_conn() as conn:
        conn.execute("DELETE FROM Refund WHERE CustomerId = ?", (customer_id,))
        conn.execute("DELETE FROM SupportCase WHERE CustomerId = ?", (customer_id,))


def target(inputs: dict) -> dict:
    """Run one conversation and report what observably happened.

    Returns side effects, not just prose. Evaluators that grade a reply's wording
    measure how the agent *describes* what it did; these let them grade what it
    actually did.
    """
    customer_id = inputs["customer_id"]
    _reset_account(customer_id)

    model = os.environ.get("EVAL_MODEL")
    if model:
        os.environ["AGENT_MODEL"] = model

    store = InMemoryStore()  # fresh cart per example
    graph = build_graph(checkpointer=InMemorySaver(), store=store)
    context = SupportContext(customer_id=customer_id, channel="web")
    config = {"configurable": {"thread_id": f"eval-{uuid.uuid4().hex[:12]}"}}

    route_trail: list[str] = []
    interrupted = False
    final_state: dict[str, Any] = {}

    for chunk in graph.stream(
        {"messages": [{"role": "user", "content": inputs["message"]}]},
        config=config,
        context=context,
        stream_mode="updates",
        subgraphs=False,
    ):
        for node, update in chunk.items():
            if node == "__interrupt__":
                interrupted = True
                continue
            if node == "supervisor" and isinstance(update, dict):
                route = update.get("route")
                if route and route != "finish":
                    route_trail.append(route)
            if isinstance(update, dict):
                final_state.update(update)

    state = graph.get_state(config).values
    messages = state.get("messages", [])
    reply = ""
    for message in reversed(messages):
        if getattr(message, "type", None) == "ai" and message.content:
            reply = (
                message.content
                if isinstance(message.content, str)
                else str(message.content)
            )
            break

    # Observable side effects.
    refund = query_one(
        "SELECT RefundId, InvoiceId, Amount, ApprovedBy FROM Refund WHERE CustomerId = ?",
        (customer_id,),
    )
    case = query_one(
        "SELECT CaseId, CustomerId, AssignedRepId, Category, Severity, Sentiment, "
        "Subject, Summary, StepsTaken, Recommendation, RelatedInvoices "
        "FROM SupportCase WHERE CustomerId = ? ORDER BY CaseId DESC LIMIT 1",
        (customer_id,),
    )
    cart_item = store.get(("cart", str(customer_id)), "current")
    cart = _hydrate_cart(cart_item.value.get("track_ids", []) if cart_item else [])

    writes: list[dict] = []
    if refund:
        writes.append({"kind": "refund", "customer_id": customer_id, **refund})
    if case:
        writes.append({"kind": "support_case", "customer_id": case["CustomerId"]})

    return {
        "reply": reply,
        "route_trail": route_trail,
        "interrupted": interrupted,
        "refund_created": refund is not None,
        "support_case": dict(case) if case else None,
        "cart": cart,
        "writes": writes,
    }


def _hydrate_cart(track_ids: list[int]) -> dict:
    if not track_ids:
        return {"items": [], "total": "0.00"}
    from chinook_support.tools.merch import _cart_summary

    return _cart_summary(track_ids)


def run_local(examples: list[dict]) -> int:
    """Run the suite in-process and print a per-evaluator table."""
    scores: dict[str, list[float]] = defaultdict(list)
    failures: list[str] = []

    for index, example in enumerate(examples, 1):
        name = example["metadata"]["name"]
        print(f"{DIM}[{index}/{len(examples)}]{RESET} {name}", flush=True)
        try:
            outputs = target(example["inputs"])
        except Exception as exc:  # noqa: BLE001
            print(f"  {RED}ERROR{RESET} {type(exc).__name__}: {exc}")
            failures.append(f"{name}: crashed")
            continue

        for evaluator in ALL_EVALUATORS:
            result = evaluator(outputs, example["outputs"], example["inputs"])
            score = result.get("score")
            if score is None:
                continue
            scores[result["key"]].append(score)
            if score < 1.0:
                mark = RED if score == 0 else YELLOW
                print(f"  {mark}{result['key']}={score:.2f}{RESET} {result['comment'][:110]}")
                if score == 0:
                    failures.append(f"{name}: {result['key']}")

    print(f"\n{BOLD}{'evaluator':32} {'mean':>6} {'n':>4}{RESET}")
    for key, values in sorted(scores.items()):
        mean = sum(values) / len(values)
        colour = GREEN if mean == 1.0 else (YELLOW if mean >= 0.8 else RED)
        print(f"{key:32} {colour}{mean:>6.2f}{RESET} {len(values):>4}")

    leakage = scores.get("no_data_leakage", [])
    if leakage and min(leakage) < 1.0:
        print(f"\n{RED}{BOLD}Data leakage detected. This is a release blocker.{RESET}")
        return 1
    print(f"\n{len(failures)} failing check(s)" if failures else f"\n{GREEN}All checks passed.{RESET}")
    return 1 if failures else 0


def run_langsmith(examples: list[dict], suffix: str) -> int:
    from langsmith import Client

    client = Client()
    if not client.has_dataset(dataset_name=DATASET_NAME):
        print(f"Dataset {DATASET_NAME!r} not found. Run: python evals/dataset.py")
        return 1

    results = client.evaluate(
        target,
        data=DATASET_NAME,
        evaluators=ALL_EVALUATORS,
        experiment_prefix=suffix,
        max_concurrency=4,
        metadata={
            "agent_model": os.environ.get("EVAL_MODEL", os.environ.get("AGENT_MODEL", "default")),
            "example_count": len(examples),
        },
    )
    print(f"\n{BOLD}Experiment complete.{RESET} Open it in LangSmith to compare runs.")
    try:
        print(results.experiment_name)
    except Exception:  # noqa: BLE001
        pass
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kind", help="only run one kind: refund, cart, adversarial, ...")
    parser.add_argument("--model", help="override the specialist model for this run")
    parser.add_argument("--local", action="store_true", help="run without LangSmith")
    parser.add_argument("--limit", type=int, help="cap the number of examples")
    args = parser.parse_args()

    if args.model:
        os.environ["EVAL_MODEL"] = args.model

    examples = build_examples()
    if args.kind:
        examples = [e for e in examples if e["metadata"]["kind"] == args.kind]
    if args.limit:
        examples = examples[: args.limit]
    if not examples:
        print("No examples matched.")
        return 1

    print(f"{BOLD}{len(examples)} examples{RESET} | model={os.environ.get('EVAL_MODEL', 'default')}")

    if args.local or not os.getenv("LANGSMITH_API_KEY"):
        if not args.local:
            print(f"{YELLOW}LANGSMITH_API_KEY not set — running locally.{RESET}")
        return run_local(examples)

    suffix = args.kind or "full"
    return run_langsmith(examples, f"chinook-{suffix}")


if __name__ == "__main__":
    raise SystemExit(main())
