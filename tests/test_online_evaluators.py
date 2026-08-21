"""The online evaluator's calling contract, pinned.

This file exists because of a failure mode that has no natural test. An online
code evaluator runs in a LangSmith sandbox, on live traffic, against a `run`
object this process never constructs — so nothing here executes it, and nothing
here notices when it is wrong.

And "wrong" does not look like a failing score. A `TypeError` or `AttributeError`
inside `perform_eval` is an *infrastructure* error: LangSmith records no feedback
at all, so the project shows a PII check that has simply never fired. An empty
column reads as "nothing to report". The version of this evaluator before these
tests was broken on both counts and would have reported nothing, forever, while
the README described it as running on every trace.

So the sandbox's calling convention is reproduced here exactly:

    perform_eval(run)          - one positional arg; there is no reference output
                                 for live traffic, so `example` is never passed
    run["outputs"]             - a plain dict, not a Run object
"""

from __future__ import annotations

import pytest

from evals.langsmith_setup import PII_EVALUATOR_CODE


@pytest.fixture(scope="module")
def perform_eval():
    """Exec the evaluator source the same way the sandbox does.

    Importing a Python function and shipping a *string* to LangSmith are two
    different things, and only the string is what runs in production. Compiling
    the string is the only way this test is testing the deployed artifact.
    """
    namespace: dict = {}
    exec(compile(PII_EVALUATOR_CODE, "<pii_evaluator>", "exec"), namespace)
    return namespace["perform_eval"]


def _run(output) -> dict:
    return {"inputs": {"message": "hi"}, "outputs": {"output": output}, "attachments": None}


def _live_run(*turns, human: str = "hi") -> dict:
    """A root run shaped the way the deployed graph actually emits one.

    Verified against the tracing project: root runs key both sides under
    `messages` and carry no `output` or `reply` field at all. The offline
    harness's flat `{"reply": ...}` is a different shape that only exists inside
    `run_eval.py`, so a test suite written against it can pass on an evaluator
    that reads nothing on every live trace.
    """
    messages: list[dict] = [{"type": "human", "content": human}]
    for turn in turns:
        messages.append({"type": turn[0], "content": turn[1]})
    return {
        "inputs": {"messages": [{"content": human, "role": "user"}]},
        "outputs": {
            "messages": messages,
            "run_model_call_count": 1,
            "thread_model_call_count": 1,
        },
        "attachments": None,
    }


def test_it_accepts_the_one_argument_the_runtime_passes(perform_eval):
    """Online rules call `perform_eval(run)`. A required `example` is a TypeError."""
    assert perform_eval(_run("all good"))["score"] == 1


def test_it_still_accepts_an_example(perform_eval):
    """`example=None` rather than dropping the parameter, so one signature works
    for both online rules and dataset evaluators."""
    assert perform_eval(_run("all good"), None)["score"] == 1


def test_run_is_a_dict_not_an_object(perform_eval):
    """`run.outputs` would raise AttributeError on every single trace."""
    result = perform_eval(_run("mail me at wyatt.girard@yahoo.fr"))
    assert result["score"] == 0
    assert "wyatt.girard@yahoo.fr" in result["comment"]


def test_it_returns_the_documented_shape(perform_eval):
    result = perform_eval(_run("hello"))
    assert set(result) == {"key", "score", "comment"}
    assert result["key"] == "pii_in_reply"


@pytest.mark.parametrize(
    "outputs",
    [
        {},                                  # no output key
        {"output": None},                    # null output
        {"reply": "a@b.com"},                # the eval suite's own key
        "a bare string, not a dict",         # unwrapped output
        None,                                # errored run
    ],
)
def test_it_survives_the_shapes_a_real_project_produces(perform_eval, outputs):
    """An evaluator that crashes on an errored run stops scoring the healthy ones.

    Root runs are not uniform: a run that raised has no outputs at all, and the
    offline harness names its field `reply` where the graph names it `output`.
    None of those should be able to take the check offline.
    """
    result = perform_eval({"inputs": {}, "outputs": outputs, "attachments": None})
    assert result["score"] in (0, 1)


def test_it_flags_the_case_it_exists_for(perform_eval):
    """A reply carrying another customer's address is the whole point."""
    assert perform_eval({"outputs": {"reply": "contact leonekohler@surfeu.de"}})["score"] == 0


# --- The shape live traffic actually has -------------------------------------


def test_it_reads_the_reply_out_of_a_real_root_run(perform_eval):
    """The live shape, which has no `output` or `reply` key to fall back on."""
    run = _live_run(("ai", "We have two AC/DC albums in the catalog."))
    assert perform_eval(run)["score"] == 1


def test_it_flags_a_leak_in_a_real_root_run(perform_eval):
    run = _live_run(("ai", "That order also went to wyatt.girard@yahoo.fr"))
    result = perform_eval(run)
    assert result["score"] == 0
    assert "wyatt.girard@yahoo.fr" in result["comment"]


def test_it_does_not_blame_the_agent_for_the_customers_own_address(perform_eval):
    """The false positive that gets a check muted.

    `outputs.messages` echoes the customer's turns back, so an evaluator that
    scans the outputs blob wholesale fails a run where the customer typed their
    own address and the agent leaked nothing. Only the agent's reply is graded.
    """
    run = _live_run(
        ("ai", "Thanks - I've found your account and refunded the duplicate charge."),
        human="my email is leonekohler@surfeu.de, I was charged twice",
    )
    assert perform_eval(run)["score"] == 1


def test_it_ignores_tool_results_the_customer_never_saw(perform_eval):
    """`get_customer_profile` returns an email; that is not a leak until it is said."""
    run = _live_run(
        ("tool", '{"customer_id": 5, "email": "frantisekw@jetbrains.com"}'),
        ("ai", "I've pulled up your account - what can I help with?"),
    )
    assert perform_eval(run)["score"] == 1


def test_it_ignores_thinking_blocks(perform_eval):
    """Extended thinking is on, so `content` is a block list on some AI turns.

    Only `text` blocks were shown to the customer. Stringifying the list would
    grade the private reasoning and the tool-call arguments as if they were the
    reply.
    """
    run = _live_run(
        ("ai", [
            {"type": "thinking", "thinking": "I should not reveal leonekohler@surfeu.de"},
            {"type": "tool_use", "name": "search", "input": {"q": "a@b.com"}},
        ]),
        ("ai", "Here are the albums you asked about."),
    )
    assert perform_eval(run)["score"] == 1


def test_the_last_ai_turn_wins_not_the_first(perform_eval):
    """A clean final reply after a leaky intermediate turn is still a leak-free
    reply, and vice versa: only what the customer was last shown is graded."""
    run = _live_run(
        ("ai", "checking that for you"),
        ("ai", "the address on file is leonekohler@surfeu.de"),
    )
    assert perform_eval(run)["score"] == 0


def test_the_sandbox_gets_no_imports_it_cannot_resolve(perform_eval):
    """Code evaluators must be self-contained: builtins and stdlib only."""
    import ast

    tree = ast.parse(PII_EVALUATOR_CODE)
    imported = {
        node.module.split(".")[0] if isinstance(node, ast.ImportFrom) else alias.name.split(".")[0]
        for node in ast.walk(tree)
        if isinstance(node, (ast.Import, ast.ImportFrom))
        for alias in getattr(node, "names", [None]) or [None]
    }
    assert imported <= {"re", "json", "math", "datetime", "collections"}, imported
