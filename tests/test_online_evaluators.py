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
