import time

import pandas as pd

from agent import run_agent

tests = [
    ("How risky is company 10?", ["get_risk_score"]),
    ("Is company 25 safe?", ["get_risk_score"]),
    ("Give me the financials of company 7.", ["lookup_company"]),
    ("What is the risk score of company 19?", ["get_risk_score"]),
    ("What does a high risk score mean?", ["find_policy"]),
    ("What should we do for a medium risk company?", ["find_policy"]),
    ("Can we rely on a score when many fields are missing?", ["find_policy"]),
    ("Is the risk score the probability of default?", ["find_policy"]),
    ("Should we approve company 14?", ["get_risk_score"]),
    ("Compare the risk of company 1 and company 10.", ["get_risk_score"]),
    ("Show profit details and risk for company 3.", ["lookup_company", "get_risk_score"]),
    ("How risky is company 99999?", ["get_risk_score"]),
    ("Who won the cricket world cup?", []),
    ("What is the weather today?", []),
    ("How risky is the company?", []),
    ("What is the policy for low risk companies?", ["find_policy"]),
    ("Tell me about company 50 and what to do next.", ["get_risk_score", "find_policy"]),
    ("Is company 2 flagged?", ["get_risk_score"]),
    ("Who makes the final decision on a loan?", ["find_policy"]),
    ("Write me a poem.", []),
]

rows, passed, counted = [], 0, 0
for q, must in tests:
    answer, steps = run_agent(q)
    called = [s.split("(")[0] for s in steps]
    if answer.startswith("Google is busy"):
        result = "SKIPPED"
    else:
        counted += 1
        ok = all(t in called for t in must) if must else len(called) == 0
        result = "PASS" if ok else "FAIL"
        passed += ok
    print(f"{result} | {q} | expected {must} | called {called}")
    rows.append({"question": q, "expected": must, "called": called,
                 "result": result, "answer": answer})
    time.sleep(8)

pd.DataFrame(rows).to_csv("eval_results.csv", index=False)
print(f"\nCorrect tool choice: {passed} of {counted} ({passed / counted:.0%})")