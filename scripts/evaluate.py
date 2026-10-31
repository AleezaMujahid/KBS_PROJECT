"""Accuracy of the engine on data/test_cases.csv.
Run from the project root:  python scripts/evaluate.py"""

import sys
sys.path.insert(0, ".")
import pandas as pd
from src.inference_engine import Engine

HEALTHY_LIMIT = 0.5   # healthy machine is a false alarm if any fault >= this

engine = Engine.from_files()
tc = pd.read_csv("data/test_cases.csv").fillna("")

rows, top1, top3, n_faulty = [], 0, 0, 0
healthy, false_alarms = 0, 0
per_fault = {}
for _, c in tc.iterrows():
    facts = {s: 1.0 for s in c["symptoms"].split(";") if s}
    ranked = engine.diagnose(facts)["results"]
    names = [r["fault"] for r in ranked]
    expected = c["expected_fault"].split("|")
    if expected == ["none"]:
        healthy += 1
        alarm = any(r["cf"] >= HEALTHY_LIMIT for r in ranked)
        false_alarms += alarm
        rows.append((c["case_id"], "none", ",".join(names[:3]) or "-", "", "", alarm))
        continue
    n_faulty += 1
    ok1 = bool(names) and names[0] in expected
    ok3 = any(n in expected for n in names[:3])
    top1 += ok1
    top3 += ok3
    for e in expected:
        d = per_fault.setdefault(e, [0, 0])
        d[0] += 1
        d[1] += ok1 and names[0] == e
    rows.append((c["case_id"], c["expected_fault"], ",".join(names[:3]) or "-", ok1, ok3, ""))

print(f"Faulty cases : {n_faulty}")
print(f"  top-1 accuracy: {top1}/{n_faulty} = {top1 / max(n_faulty, 1):.1%}")
print(f"  top-3 accuracy: {top3}/{n_faulty} = {top3 / max(n_faulty, 1):.1%}")
print(f"Healthy cases: {healthy}")
print(f"  false alarms (fault conf >= {HEALTHY_LIMIT}): {false_alarms}/{healthy}")
print("Per fault (top-1 hits / cases containing that fault):")
for k, (n, h) in sorted(per_fault.items()):
    print(f"  {k}: {h}/{n}")
pd.DataFrame(rows, columns=["case_id", "expected", "engine_top3", "top1_ok",
                            "top3_ok", "false_alarm"]).to_csv(
    "data/evaluation_results.csv", index=False)
print("Details saved to data/evaluation_results.csv")
