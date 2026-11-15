"""
Prompt / agent evaluation: a fixed test set run after every prompt change (regression testing for LLMs).
Each case checks: did the agent use the expected tool?  does the answer contain the expected facts?
Run: python -m eval.run_eval      -> prints a score; change prompts.py and compare.
"""
import json
from pathlib import Path

from app import rag
from app.agent import PlantCopilot
from app.prompts import PROMPT_VERSION

CASES = json.loads((Path(__file__).parent / "cases.json").read_text(encoding="utf-8"))


def main():
    rag.build_index()
    copilot = PlantCopilot()
    passed = 0
    for c in CASES:
        r = copilot.ask(c["question"])
        tool_ok = set(c["expected_tools"]) <= set(r.tools_used)
        text = r.answer.lower().replace(",", "")
        facts_ok = all(f.lower() in text for f in c["must_contain"])
        if c.get("must_contain_any"):  # e.g. the agent must admit it does not know
            facts_ok = facts_ok and any(f in text for f in c["must_contain_any"])
        ok = tool_ok and facts_ok
        passed += ok
        print(f"[{'PASS' if ok else 'FAIL'}] {c['id']:<16} tools={r.tools_used}"
              f"{'' if tool_ok else ' (wrong tools)'}{'' if facts_ok else ' (missing expected facts)'}")
    print(f"\nPrompt v{PROMPT_VERSION}: {passed}/{len(CASES)} passed ({passed / len(CASES):.0%})")


if __name__ == "__main__":
    main()
