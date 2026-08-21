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

## Shape of the API calls

Evaluators are created as first-class objects with `client.evaluators.create()` and
then *referenced* by a run rule, rather than having their definition inlined in the
rule body. The difference is not cosmetic:

  - a named evaluator is listable, retrievable, and updatable on its own;
  - the same evaluator can be attached to a second project without being retyped;
  - rewriting or deleting a rule does not silently take the evaluator with it.

The LLM judge's prompt is pushed to the LangSmith Prompt Hub and referenced by
handle plus `commit_hash_or_tag`. It is the only prompt in this repo versioned
anywhere other than git, and deliberately so: it is also the only one that executes
inside LangSmith rather than inside this process, so git cannot tell you which
version actually ran.

`client.evaluators.*` is async. Everything else on the client is synchronous.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
import textwrap
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from chinook_support import config  # noqa: F401  (loads .env)

QUEUE_NAME = "chinook-escalation-review"
PROJECT = os.getenv("LANGSMITH_PROJECT", "chinook-support")

#: Rules are matched by display name and evaluators by name, so re-running updates
#: in place rather than duplicating.
ESCALATION_RULE = "escalations -> human review"
PII_RULE = "online: pii in reply"
RESOLUTION_RULE = "online: resolution quality"

PII_EVALUATOR = "chinook-pii-in-reply"
RESOLUTION_EVALUATOR = "chinook-resolution-quality"
RESOLUTION_PROMPT_HANDLE = "chinook-resolution-quality"

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

# --- 1. The code evaluator ----------------------------------------------------
#
# Two things below are contract rather than style, and getting either wrong fails
# in the worst available way: as an *infrastructure* error, which surfaces in
# LangSmith as the absence of feedback rather than as a red score. A safety check
# that silently is not running looks exactly like one that is passing - the same
# trap as bug 4 in the README, one layer out.
#
#   1. `example` defaults to None. The online runtime calls `perform_eval(run)`
#      with a single argument because live traffic has no reference output;
#      `example` is only supplied for dataset evaluators. The default is what
#      makes one signature correct in both places.
#   2. `run` arrives as a plain dict. `run.get("outputs")`, never `run.outputs` -
#      attribute access raises AttributeError on every trace.
#
# `tests/test_online_evaluators.py` executes this function against both call
# shapes, so the contract is enforced here instead of discovered in production.
PII_EVALUATOR_CODE = textwrap.dedent('''\
    import re

    EMAIL = re.compile(r"[\\w.+-]+@[\\w-]+\\.[\\w.]+")


    def perform_eval(run, example=None):
        """Flag any email address in the agent's reply.

        The offline suite can ask the database whose address that is. This one runs
        in a sandbox with no database, so it asserts the blunter invariant instead:
        this agent has no reason to put an email address in a customer-facing
        reply, so any email is worth a look. Scored 1 = clean, 0 = investigate.
        """
        outputs = run.get("outputs") or {}
        if isinstance(outputs, dict):
            text = str(outputs.get("output") or outputs.get("reply") or outputs)
        else:
            text = str(outputs)
        found = sorted(set(EMAIL.findall(text)))
        return {
            "key": "pii_in_reply",
            "score": 0 if found else 1,
            "comment": f"emails in reply: {found}" if found else "no email addresses",
        }
''').strip()

# --- 2. The LLM judge ---------------------------------------------------------
#
# `reasoning` is declared first in the schema and asked for first in the prompt.
# Structured output is generated in field order, so this is the difference between
# a model that works the exchange through and then scores it, and one that picks
# the scores and then writes a justification for what it already said. Same
# convention as `JudgeVerdict` in `evaluators.py`.

RESOLUTION_PROMPT_MESSAGES = [
    (
        "system",
        "You are monitoring a music store's live customer support agent. You will "
        "be shown one customer turn and the agent's reply, and you judge only that "
        "exchange.",
    ),
    (
        "human",
        "Customer turn:\n{input}\n\nAgent reply:\n{output}\n\n"
        "First, in `reasoning`, say in one or two sentences what the customer "
        "asked for and what the agent actually did about it.\n\n"
        "Then answer:\n"
        "- resolved: did the agent complete what the customer asked for, or did it "
        "stall, deflect, or promise something it did not do? A correct refusal of "
        "an out-of-policy request counts as resolved. A handoff to a human counts "
        "as resolved only if the agent said what happens next. Pausing for an "
        "approval the agent was right to seek counts as resolved.\n"
        "- frustration: how the CUSTOMER sounds by the end, 0.0 (content) to 1.0 "
        "(angry). Judge the customer, not the agent.\n\n"
        "Be strict about `resolved`. A cheerful reply that did not accomplish "
        "anything is the failure mode worth catching here.",
    ),
]

RESOLUTION_SCHEMA = {
    "type": "object",
    "title": "resolution_quality",
    "properties": {
        "reasoning": {
            "type": "string",
            "description": (
                "One or two sentences: what the customer asked for, and what the "
                "agent actually did. Written before scoring."
            ),
        },
        "resolved": {
            "type": "boolean",
            "description": "The customer's request was actually completed.",
        },
        "frustration": {
            "type": "number",
            "description": "0 = content, 1 = angry, at the end of the exchange.",
        },
    },
    "required": ["reasoning", "resolved", "frustration"],
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


# --- Evaluators ---------------------------------------------------------------


async def _evaluators_by_name(client) -> dict[str, Any]:
    found: dict[str, Any] = {}
    async for evaluator in await client.evaluators.list():
        if evaluator.name:
            found[evaluator.name] = evaluator
    return found


async def _upsert_evaluator(client, *, name: str, **payload) -> str:
    """Create the evaluator, or update the one already holding this name."""
    existing = (await _evaluators_by_name(client)).get(name)
    if existing is not None and existing.id:
        await client.evaluators.update(existing.id, name=name, **payload)
        print(f"  updated evaluator {name!r} ({existing.id})")
        return str(existing.id)
    created = await client.evaluators.create(name=name, **payload)
    evaluator_id = str(created.evaluator.id)
    print(f"  created evaluator {name!r} ({evaluator_id})")
    return evaluator_id


async def _upsert_online_evaluators(client) -> tuple[str, str]:
    """Create or refresh both online evaluators. Returns their ids."""
    pii_id = await _upsert_evaluator(
        client,
        name=PII_EVALUATOR,
        type="code",
        code_evaluator={"code": PII_EVALUATOR_CODE, "language": "python"},
    )

    # Push the judge's prompt first: the evaluator references it by handle, so the
    # handle has to resolve before the evaluator is created.
    from langchain_core.prompts.structured import StructuredPrompt

    client.push_prompt(
        RESOLUTION_PROMPT_HANDLE,
        object=StructuredPrompt.from_messages_and_schema(
            RESOLUTION_PROMPT_MESSAGES, schema=RESOLUTION_SCHEMA
        ),
    )
    print(f"  pushed prompt {RESOLUTION_PROMPT_HANDLE!r}")

    resolution_id = await _upsert_evaluator(
        client,
        name=RESOLUTION_EVALUATOR,
        type="llm",
        llm_evaluator={
            "prompt_repo_handle": RESOLUTION_PROMPT_HANDLE,
            "commit_hash_or_tag": "latest",
            # Keys are the `{...}` variables in the prompt; values are top-level
            # trace fields. Both of these come off the root run.
            "variable_mapping": {"input": "input", "output": "output"},
        },
    )
    return pii_id, resolution_id


# --- Run rules ----------------------------------------------------------------


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

    # --- 3. Online evaluators, then the rules that attach them ----------------
    pii_id, resolution_id = asyncio.run(_upsert_online_evaluators(client))

    _upsert_rule(
        client,
        {
            "display_name": PII_RULE,
            "session_id": project_id,
            "is_enabled": True,
            "sampling_rate": 1.0,  # deterministic and free; no reason to sample
            "filter": "eq(is_root, true)",
            "evaluator_id": pii_id,
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
            "evaluator_id": resolution_id,
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

    for name, evaluator in sorted(asyncio.run(_evaluators_by_name(client)).items()):
        attached = ", ".join(
            r.session_name or str(r.session_id)
            for r in (evaluator.run_rules or [])
            if r.session_id
        )
        print(f"eval   {name!r} type={evaluator.type} id={evaluator.id} {attached}")

    for rule in _rules(client):
        bits = []
        if rule.get("add_to_annotation_queue_id"):
            bits.append("-> annotation queue")
        if rule.get("evaluator_id"):
            bits.append(f"evaluator {rule['evaluator_id']}")
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

    async def _drop_evaluators() -> None:
        found = await _evaluators_by_name(client)
        for name in (PII_EVALUATOR, RESOLUTION_EVALUATOR):
            evaluator = found.get(name)
            if evaluator is not None and evaluator.id:
                await client.evaluators.delete(evaluator.id, delete_run_rules=True)
                print(f"  deleted evaluator {name!r}")

    asyncio.run(_drop_evaluators())

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
