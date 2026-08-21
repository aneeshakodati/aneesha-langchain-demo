"""Create the LangSmith production-monitoring setup, in code.

    python evals/langsmith_setup.py            # create or update everything
    python evals/langsmith_setup.py --show     # print what exists, change nothing
    python evals/langsmith_setup.py --delete   # tear it down

Annotation queues and automation rules are normally clicked together in the
LangSmith UI, which makes them invisible to code review, impossible to diff, and
untransferable to a second workspace. They are configuration, so they belong in
the repo with everything else. This script is idempotent: run it twice and the
second run updates rather than duplicating.

Three things get created against the live tracing project:

1. `chinook-escalation-review` - an annotation queue with a rubric. Every trace
   that filed a support case lands here for the supervisor who owns the handoff to
   grade. Their scores are the seed corpus for new eval cases: production ->
   dataset, which is the loop the README argues for.

2. `pii_in_reply` - a code evaluator on every traced run. Cheap, deterministic,
   no model call. The suite's `no_data_leakage` evaluator can query Chinook and so
   knows exactly whose email is whose; a sandboxed online evaluator cannot, so this
   one asserts the weaker but still useful invariant that the agent has no business
   emitting *any* email address at all. Narrow and true beats broad and noisy - an
   online check that cries wolf gets muted within a week.

3. `resolution_quality` - an LLM judge sampled at 20% of live traffic. This is the
   question offline evals structurally cannot answer, because it is about the
   traffic you actually got rather than the cases you thought to write down: are
   real customers leaving satisfied?

The rules attach to `LANGSMITH_PROJECT`, so they watch the demo and Studio traffic,
not the eval experiments - grading your own eval runs online double-counts them.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from chinook_support import config  # noqa: F401  (loads .env)

QUEUE_NAME = "chinook-escalation-review"
PROJECT = os.getenv("LANGSMITH_PROJECT", "chinook-support")

#: Rules are matched by name so re-running updates in place.
ESCALATION_RULE = "escalations -> human review"
PII_RULE = "online: pii in reply"
RESOLUTION_RULE = "online: resolution quality"

# The queue's rubric. These are the questions the supervisor is actually being
# asked, and they deliberately mirror the offline judge's criteria in
# `evaluators.py` - that overlap is the point. Where the human and the judge
# disagree, the judge is the thing that needs fixing.
RUBRIC_INSTRUCTIONS = (
    "You are the support representative this ticket was handed to. Grade it as "
    "the person who has to act on it, not as an editor. The question is always "
    "'could I work this case without reading the transcript or calling the "
    "customer back?'"
)
RUBRIC_ITEMS: list[dict[str, Any]] = [
    {
        "feedback_key": "ticket_actionable",
        "description": "Could you act on this without reading the conversation?",
        "is_required": True,
    },
    {
        "feedback_key": "ticket_accurate",
        "description": "Are the order numbers, amounts and dates correct?",
        "is_required": True,
    },
    {
        "feedback_key": "severity_correct",
        "description": "Is the severity right, or would you have triaged it differently?",
        "is_required": False,
    },
]

PII_EVALUATOR_CODE = '''
import re

EMAIL = re.compile(r"[\\w.+-]+@[\\w-]+\\.[\\w.]+")


def perform_eval(run, example):
    """Flag any email address in the agent's reply.

    The offline suite can ask the database whose address that is. This one runs in
    a sandbox with no database, so it asserts the blunter invariant instead: this
    agent has no reason to put an email address in a customer-facing reply, so any
    email is worth a look. Scored 1 = clean, 0 = investigate.
    """
    text = str((run.outputs or {}).get("output") or run.outputs or "")
    found = EMAIL.findall(text)
    return {
        "key": "pii_in_reply",
        "score": 0 if found else 1,
        "comment": f"emails in reply: {found}" if found else "no email addresses",
    }
'''.strip()

RESOLUTION_PROMPT = """\
You are monitoring a music store's live customer support agent.

Customer turn and agent reply:
{{input}}
{{output}}

Answer two things about this exchange:

- resolved: did the agent actually complete what the customer asked for, or did it
  stall, deflect, or promise something it did not do? A correct refusal of an
  out-of-policy request counts as resolved. A handoff to a human counts as
  resolved only if the agent said what happens next.
- frustration: how the CUSTOMER sounds by the end, 0 (content) to 1 (angry).

Be strict about `resolved`. A cheerful reply that did not accomplish anything is
the failure mode worth catching here.
"""

RESOLUTION_SCHEMA = {
    "type": "object",
    "title": "resolution_quality",
    "properties": {
        "resolved": {
            "type": "boolean",
            "description": "The customer's request was actually completed.",
        },
        "frustration": {
            "type": "number",
            "description": "0 = content, 1 = angry, at the end of the exchange.",
        },
        "comment": {"type": "string", "description": "One sentence of reasoning."},
    },
    "required": ["resolved", "frustration", "comment"],
}


def _client():
    from langsmith import Client

    return Client()


def _host(client) -> str:
    return getattr(client, "_host_url", None) or "https://smith.langchain.com"


def _project_id(client) -> str | None:
    for project in client.list_projects(name=PROJECT):
        return str(project.id)
    return None


def _rules(client) -> list[dict]:
    return client.request_with_retries("GET", "/runs/rules").json()


def _upsert_rule(client, payload: dict) -> None:
    """Create the rule, or PATCH the existing one with the same display name."""
    existing = {r["display_name"]: r for r in _rules(client)}
    name = payload["display_name"]
    if name in existing:
        response = client.request_with_retries(
            "PATCH", f"/runs/rules/{existing[name]['id']}", json=payload
        )
        verb = "updated"
    else:
        response = client.request_with_retries("POST", "/runs/rules", json=payload)
        verb = "created"
    if response.status_code >= 400:
        print(
            f"  FAILED to {verb[:-1]} {name!r}: {response.status_code} {response.text[:400]}"
        )
        return
    print(f"  {verb} rule {name!r}")


def setup() -> int:
    client = _client()

    project_id = _project_id(client)
    if project_id is None:
        print(
            f"Tracing project {PROJECT!r} does not exist yet — the rules have "
            f"nothing to attach to.\nRun `make demo` once (with LANGSMITH_TRACING=true) "
            f"and then re-run this."
        )
        return 1

    # --- 1. Annotation queue -------------------------------------------------
    queues = {q.name: q for q in client.list_annotation_queues(name=QUEUE_NAME)}
    if QUEUE_NAME in queues:
        queue = queues[QUEUE_NAME]
        client.update_annotation_queue(
            queue.id,
            name=QUEUE_NAME,
            description="Escalation tickets awaiting supervisor review.",
        )
        print(f"  queue {QUEUE_NAME!r} already exists")
    else:
        queue = client.create_annotation_queue(
            name=QUEUE_NAME,
            description="Escalation tickets awaiting supervisor review.",
            rubric_instructions=RUBRIC_INSTRUCTIONS,
            rubric_items=RUBRIC_ITEMS,
        )
        print(f"  created queue {QUEUE_NAME!r}")

    # --- 2. Route escalations into it ----------------------------------------
    # `tree_filter` matches on any run in the trace, which is how "this
    # conversation filed a ticket" is expressed: the root run's own output says
    # nothing about it, but a `file_escalation` tool span in the tree does.
    _upsert_rule(
        client,
        {
            "display_name": ESCALATION_RULE,
            "session_id": project_id,
            "is_enabled": True,
            "sampling_rate": 1.0,
            "filter": "eq(is_root, true)",
            "tree_filter": 'eq(name, "file_escalation")',
            "add_to_annotation_queue_id": str(queue.id),
        },
    )

    # --- 3. Online evaluators -------------------------------------------------
    _upsert_rule(
        client,
        {
            "display_name": PII_RULE,
            "session_id": project_id,
            "is_enabled": True,
            "sampling_rate": 1.0,  # deterministic and free; no reason to sample
            "filter": "eq(is_root, true)",
            "code_evaluators": [{"code": PII_EVALUATOR_CODE, "language": "python"}],
        },
    )
    _upsert_rule(
        client,
        {
            "display_name": RESOLUTION_RULE,
            "session_id": project_id,
            "is_enabled": True,
            "sampling_rate": 0.2,  # a model call per trace; sample it
            "filter": "eq(is_root, true)",
            "evaluators": [
                {
                    "structured": {
                        "prompt": [["human", RESOLUTION_PROMPT]],
                        "template_format": "mustache",
                        "schema": RESOLUTION_SCHEMA,
                        "variable_mapping": {
                            "input": "input",
                            "output": "output",
                        },
                        "model": {"provider": "anthropic", "name": "claude-sonnet-5"},
                    }
                }
            ],
        },
    )

    print(f"\n  {_host(client)}/o/-/projects/p/{project_id}?tab=rules")
    print(f"  {_host(client)}/annotation-queues/{queue.id}")
    return 0


def show() -> int:
    client = _client()
    project_id = _project_id(client)
    print(f"project {PROJECT!r}: {project_id or 'DOES NOT EXIST'}")
    for queue in client.list_annotation_queues():
        print(f"queue  {queue.name}  ({queue.id})")
    for rule in _rules(client):
        bits = []
        if rule.get("add_to_annotation_queue_id"):
            bits.append("-> annotation queue")
        if rule.get("code_evaluators"):
            bits.append(f"{len(rule['code_evaluators'])} code evaluator(s)")
        if rule.get("evaluators"):
            bits.append(f"{len(rule['evaluators'])} llm evaluator(s)")
        print(
            f"rule   {rule['display_name']!r} enabled={rule.get('is_enabled')} "
            f"sampling={rule.get('sampling_rate')} {', '.join(bits)}"
        )
    return 0


def delete() -> int:
    client = _client()
    for rule in _rules(client):
        if rule["display_name"] in (ESCALATION_RULE, PII_RULE, RESOLUTION_RULE):
            client.request_with_retries("DELETE", f"/runs/rules/{rule['id']}")
            print(f"  deleted rule {rule['display_name']!r}")
    for queue in client.list_annotation_queues(name=QUEUE_NAME):
        client.delete_annotation_queue(queue.id)
        print(f"  deleted queue {queue.name!r}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--show", action="store_true", help="print current state only")
    parser.add_argument("--delete", action="store_true", help="tear it all down")
    args = parser.parse_args()

    if not os.getenv("LANGSMITH_API_KEY"):
        print("LANGSMITH_API_KEY is not set.")
        return 1
    if args.show:
        return show()
    if args.delete:
        return delete()
    return setup()


if __name__ == "__main__":
    raise SystemExit(main())
