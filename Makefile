.PHONY: setup db reset studio demo test eval eval-local dataset clean

VENV := .venv
PY   := $(VENV)/bin/python

setup:  ## create the venv, install everything, build the databases
	uv venv --python 3.13 $(VENV)
	uv pip install --python $(PY) -e ".[dev]"
	$(PY) scripts/build_db.py
	@test -f .env || (cp .env.example .env && echo "Created .env — add your ANTHROPIC_API_KEY")

db:  ## build the Chinook databases from the upstream dump
	$(PY) scripts/build_db.py

reset:  ## wipe demo state: refunds, cases, orders, carts, conversations
	$(PY) scripts/reset_demo.py

studio:  ## run LangGraph Studio
	@# --allow-blocking is required: the tools use synchronous sqlite3, and the dev
	@# server's blocking-call detector rejects that on the event loop. Local SQLite
	@# reads are sub-millisecond; a production deployment would use an async driver.
	$(VENV)/bin/langgraph dev --allow-blocking

demo:  ## run the scripted 7-act demo (resets state first)
	$(PY) demo.py

test:  ## unit tests — policy engine, cart solver, tenant isolation
	$(PY) -m pytest tests/ -q

dataset:  ## push the eval dataset to LangSmith
	$(PY) evals/dataset.py

eval:  ## run the eval suite as a LangSmith experiment
	$(PY) evals/run_eval.py

eval-local:  ## run the same evaluators without LangSmith
	$(PY) evals/run_eval.py --local

clean:
	rm -rf .langgraph_api **/__pycache__ .pytest_cache *.egg-info
