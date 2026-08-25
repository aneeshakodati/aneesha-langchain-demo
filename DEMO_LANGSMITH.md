# LangSmith Demo Script — Chinook Support Agent

A presenter's cue sheet for selling **LangSmith** specifically. `DEMO.md` is the
longer product walkthrough — architecture, the customer journey, why the agent is
shaped the way it is. This one inverts the emphasis: the agent is a realistic trace
generator, and LangSmith is the thing being bought.

Bullets are what to *do* and what the point *is*. They are not lines to read aloud.

Every number below was measured in this repository. Where something is designed but
not demonstrated, it says so.

- **Full run:** ~30 minutes.
- **Short version:** §2, §4 (run 6 only), §5, §6. See *Cut to 15 minutes*.
- **The two beats that carry the whole demo:** run 6 in §4, and the monitoring
  stories in §6. Everything else is supporting material.

---

## Pre-flight

Do this before they join, not while they watch.

- `make reset` — a clean demo database. Skip it and order 413 will not auto-refund,
  because a previous run already refunded it.
- `make studio` in a terminal. `--allow-blocking` is in the Makefile target; the
  tools are synchronous SQLite and the dev server rejects that without it.
- Three LangSmith tabs open: the **dataset comparison view**, the
  **`chinook-support` project**, and the **annotation queue**.
- Confirm the workspace has an `ANTHROPIC_API_KEY` under **Settings → Secrets**. The
  online resolution judge executes inside LangSmith and cannot read your `.env`.
  Without it, it records auth errors where scores should be — which in the UI looks
  like a judge that ran and had nothing to say.

---

## 1. Frame the problem — 2 min, no tool on screen

- Support agent for a music store: refunds, cart building, escalation to a human.
- The hard part is not making it work once. It is knowing it still works after the
  next prompt edit.
- Three things that must never happen: leak another customer's data, refund without
  approval, loop forever.
- **Ask them:** *"How do you currently know your agent didn't get worse this week?"*
  Most have no answer. That gap is what the rest of the demo fills.

## 2. One trace — observability — 4 min

- In Studio, as customer 1: *"I'd like a refund on order 414 — the tracks won't
  play."*
- Watch control move: `supervisor → billing → supervisor`.
- The approval interrupt fires. $25.74 is over the auto-approve limit, so a human
  has to sign off.
- Switch to the trace in LangSmith.
- **Point at:** the tool-call tree, per-node latency, token counts, and the exact
  prompt that went to the model.
- **The point:** this cost one environment variable. There is no instrumentation
  code in the agent.
- **Then the isolation beat:** change `customer_id` to 2 in the run config, re-ask
  "what did I buy?". Different account, no code change. Identity lives in runtime
  context, so there is no token the model can emit to become someone else.

## 3. The dataset — evals as regression tests — 4 min

- Open the `chinook-support-agent` dataset. 35 examples.
- **Point at:** eight adversarial cases sitting in the *main* dataset, not in a
  separate security suite run occasionally. Every experiment measures them.
- **Say this explicitly, it is the most transferable idea in the repo:** the expected
  refund decisions are *computed* at build time by calling the same policy engine the
  agent uses. They are not typed into a file. Hand-written expectations rot the
  moment a threshold changes, and a stale reference gives you a suite that
  confidently fails correct behaviour.
- Five evaluators: four deterministic — leakage, policy, cart constraints, routing —
  and one LLM judge.
- **The line:** *"A suite made entirely of LLM judges measures whether one model
  agrees with another. The judge earns its place here on exactly one thing: whether
  an escalation summary is any good."*

## 4. The comparison view — the money slide — 8 min

This is the centre of the demo. Do not rush it. Nine experiments side by side.

**Run 1 → 2 — the tests were wrong before the agent was**

- Carts 0.50, judge 0.69. Both were evaluator bugs, not agent bugs.
- The cart checker compared track genres to the literal word the customer typed. The
  catalog has both `Rock` and `Rock And Roll`, and the system deliberately expands to
  the adjacent subgenre.
- The judge was asked whether order totals and dates were invented, having never been
  shown the tool calls they came from.

**Run 3 — the regression (`chinook-full-1edfed31`)**

- Three lines added to the router prompt to fix one mis-routed example.
- Cost four other routes. Route accuracy 0.96 → 0.85, carts 1.00 → 0.67.
- **Say it plainly:** a tweak worth one point took away four, and nothing but this
  diff would have shown it.
- Reverted in run 4. The original 1-in-27 mis-route is still there and is the better
  trade.

**Run 6 — the one that sells the product (`chinook-full-50137e33`)**

- 11 of 35 examples crashed. Route accuracy 0.70, judge 0.00.
- The cause was a *safety rail* added to prevent runaway loops: a recursion limit
  of 15.
- Correct arithmetic on the wrong graph. It counted supersteps in the parent, but the
  limit propagates into subgraphs, each counts its own, and the escalation specialist
  needs about 51.
- **The kicker, delivered slowly:** 74 unit tests passed — including one asserting
  the ceiling was fine that never instantiated a specialist. A clean `demo.py` run. A
  reviewed diff. All green.
- **Land it:** *"Only the eval run caught this. Everything else told us we were
  fine."*

**Runs 7-8 — recovery**

- Ceiling now derived from the compiled agents, so adding a middleware fails a test
  instead of a third of the dataset. Everything back to 1.00.

## 5. Cost — "can we afford the cheap model?" — 4 min

- Run 9 is the identical suite on Haiku. One CLI flag: `make eval-haiku`.

| | Sonnet (run 8) | Haiku (run 9) |
|---|---|---|
| `no_data_leakage` | 1.00 | 0.97 |
| `policy_adherence` | 1.00 | 1.00 |
| `cart_constraints_satisfied` | 1.00 | 1.00 |
| `route_accuracy` | 1.00 | 1.00 |
| median latency | 9.0s | 5.1s |
| total tokens | 408,551 | 317,867 |

- 43% faster, 22% cheaper in tokens, and identical on routing, carts and policy —
  the three things a support agent does most.
- **Deliver this caveat out loud. It will be misheard otherwise.** Haiku did *not*
  leak customer data. It called no tools at all, touched no store data, and refused
  correctly — then said the claimed name twice while refusing, which the deliberately
  strict evaluator counts as a leak.
- **The real finding:** the deterministic evaluators did not move. What moved was the
  behaviour held in place by *instructions*. A prompt-level mitigation does not
  transfer down a model tier, and only a suite tells you that.
- **Their takeaway:** this is a procurement decision made with a table in front of
  you, not by intuition.

## 6. Production monitoring — 5 min

This is the half that renews the contract, and the half most demos skip.

- Open the `chinook-support` project, filtered to live traces.
- **Online PII evaluator:** scored 11 of 11 live traces. No sampling — it is
  deterministic and free.
- **Tell the bug story.** This check was attached at 100% sampling and had never run.
  LangSmith calls a code evaluator with `run` as a plain dict; it was written to take
  two arguments and read attributes. Every invocation raised `TypeError`. A crashed
  evaluator records *no feedback at all* rather than a red score, so the column was
  simply empty — and an empty column reads as "nothing to report".
- **Resolution judge:** executes inside LangSmith, sampled at 20% because it bills a
  model call per trace.
- Open the **annotation queue**. Only conversations that actually filed an escalation.
- **Second infrastructure story.** This queue held 461 items. The rule matched any
  root run, and the eval suite's own evaluators are root runs in the same project —
  450 of the 461 were evaluator invocations. A review queue a human cannot face is
  the same as no review queue, and it fails quietly: the rule looks healthy and the
  supervisor just stops opening it.
- **Close the loop:** human review → feedback → becomes a dataset example → shows up
  in the next experiment.

## 7. Close — 3 min

- Nine experiments, and **every number that moved was a bug we did not know about.**
- Of the fifteen changes the suite drove, **eight were to the evaluators or the
  monitoring itself.**
- Four distinct ways a control looked healthy while doing nothing:
  - the leakage check reading 1.00 while scanning a base64 thinking blob
  - the PII monitor showing an empty column while crashing on every trace
  - the review queue looking busy while holding 11 real items out of 461
  - the leakage tripwire answering from whichever database happened to open first
- **The closing line:** *"A green check is only as trustworthy as the evidence that
  it ran. That is what you are buying."*
- **Ask:** *"Which of your agents has a number attached to it right now?"*

---

## Cut to 15 minutes

Keep §2 (one trace), §4 **run 6 only**, §5 (the cost table), §6 (monitoring). Drop
§1, §3, and runs 1-3. Run 6 plus the Haiku table is the entire argument.

## The two objections you will get

- **"Couldn't better unit tests have caught run 6?"** — There was one. It asserted
  `11 <= 15 < 22` and never instantiated a specialist. The wrong measurement written
  down twice, in the constant and in the test guarding it, which is what made the bug
  feel covered.
- **"Isn't the LLM judge marking its own homework?"** — Four of the five evaluators
  involve no model at all; they assert against the policy engine and the cart solver.
  The judge grades one subjective thing, and it is the score that has been wrong most
  often.

## Known rough edges — do not get caught out

- The escalation judge sits at 0.91 (Sonnet) / 0.88 (Haiku), and every deduction is
  on a ticket filed in response to an *adversarial* prompt. Those tickets are thin
  because there is little to write when the "issue" is someone probing for another
  account. It is an open scoping question about what the rubric should grade, not a
  quality regression. Say so before they find it.
- `refund-duplicate-charge` fails roughly 1 in 43 runs. If it is on screen and picks
  the wrong branch, you are explaining a known flake live. Consider not demoing that
  case.
- `tests/` reads the shared demo database. Run `demo.py` and then `pytest` and two
  policy tests fail until `make reset`.
