"""Offline smoke test: exercises the graph wiring with a stubbed model.

No network calls, no API key required. Verifies llm_call -> tool_node ->
llm_call -> END routing and that the tool actually runs.
"""

from langchain.messages import AIMessage, HumanMessage

import agent as A


class StubModel:
    def __init__(self):
        self.n = 0

    def invoke(self, messages):
        self.n += 1
        if self.n == 1:
            return AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "add",
                        "args": {"a": 3, "b": 4},
                        "id": "call_1",
                        "type": "tool_call",
                    }
                ],
            )
        return AIMessage(content="3 + 4 = 7")


A.model_with_tools = StubModel()

out = A.agent.invoke({"messages": [HumanMessage(content="Add 3 and 4.")]})
for m in out["messages"]:
    m.pretty_print()
print("llm_calls:", out["llm_calls"])
