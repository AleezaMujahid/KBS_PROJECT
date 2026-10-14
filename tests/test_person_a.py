"""Person A's checks. Run from the project root:  python tests/test_person_a.py
Every check prints PASS or stops with a clear message."""
import json
import sys
sys.path.insert(0, ".")
import pandas as pd
from src.preprocess import load_data, row_to_symptoms

MODES = ["TWF", "HDF", "PWF", "OSF"]
ok = lambda msg: print("PASS ", msg)

# 1. data loads cleanly ---------------------------------------------------
df = load_data("data/ai4i2020.csv")
assert len(df) == 10000, f"expected 10000 rows, got {len(df)}"
assert df.isna().sum().sum() == 0, "dataset has missing values"
ok("dataset: 10000 rows, no missing values")

# 2. thresholds reproduce the labels exactly ------------------------------
sym = df.apply(row_to_symptoms, axis=1)
has = lambda name: sym.apply(lambda s: name in s)
hdf = has("low_temp_diff") & has("low_speed")
pwf = has("low_power") | has("high_power")
osf = has("high_strain")
for name, mask, col in [("HDF", hdf, "HDF"), ("PWF", pwf, "PWF"), ("OSF", osf, "OSF")]:
    assert (mask == (df[col] == 1)).all(), f"{name}: symptom rule disagrees with labels"
    ok(f"{name}: symptom rule matches all {int(df[col].sum())} labelled failures, no false alarms")
wear = has("high_tool_wear")
assert (df[df["TWF"] == 1]["wear"] >= 198).all()
ok(f"TWF: high_tool_wear fires on {int(wear.sum())} rows, {int((wear & (df.TWF==1)).sum())} are real failures (weak evidence, as expected)")

# 3. knowledge base is self-consistent ------------------------------------
kb = json.load(open("data/knowledge_base.json"))
rules = json.load(open("data/rules.json"))
sym_ids = {s["id"] for s in kb["symptoms"]}
fault_ids = {f["id"] for f in kb["faults"]}
action_ids = {a["id"] for a in kb["actions"]}
for r in rules:
    assert set(r["if"]) <= sym_ids, f'{r["id"]} uses an unknown symptom: {set(r["if"]) - sym_ids}'
    assert r["then"] in fault_ids, f'{r["id"]} points to unknown fault {r["then"]}'
    assert 0 < r["cf"] <= 1, f'{r["id"]} confidence must be in (0, 1]'
    assert r.get("source"), f'{r["id"]} has no source'
for f in kb["faults"]:
    assert set(f["actions"]) <= action_ids, f'{f["id"]} uses an unknown action'
for a in kb["actions"]:
    for k in ["cost", "downtime_hrs", "safety_benefit", "production_impact"]:
        assert isinstance(a[k], (int, float)), f'{a["id"]}.{k} must be a number'
    for k in ["cost", "safety_benefit", "production_impact"]:
        assert 1 <= a[k] <= 5, f'{a["id"]}.{k} must be 1 to 5'
ids = [r["id"] for r in rules]
assert len(ids) == len(set(ids)), "duplicate rule ids"
ok(f"knowledge base: {len(sym_ids)} symptoms, {len(fault_ids)} faults, {len(action_ids)} actions, {len(rules)} rules, all references valid")

# 4. test_cases.csv is well formed and labelled from the data -------------
tc = pd.read_csv("data/test_cases.csv").fillna("")
assert len(tc) >= 60, "too few test cases"
assert tc["case_id"].is_unique
for _, c in tc.iterrows():
    for s in filter(None, c["symptoms"].split(";")):
        assert s in sym_ids, f'{c["case_id"]}: unknown symptom {s}'
    for f in c["expected_fault"].split("|"):
        assert f == "none" or f in fault_ids, f'{c["case_id"]}: unknown fault {f}'
ok(f"test_cases.csv: {len(tc)} cases, all symptoms and faults valid")

# 5. quick sanity run of the rules (a simple stand-in for Person B's engine)
def diagnose(symptoms):
    result = {}
    for r in rules:
        if set(r["if"]) <= symptoms:
            old = result.get(r["then"], 0.0)
            result[r["then"]] = old + r["cf"] * (1 - old)     # combine confidences
    return result

hits = total = 0
false_alarms = healthy = 0
for _, c in tc.iterrows():
    res = diagnose(set(filter(None, c["symptoms"].split(";"))))
    if c["expected_fault"] == "none":
        healthy += 1
        false_alarms += any(v >= 0.5 for v in res.values())
    else:
        total += 1
        top = max(res, key=res.get) if res else None
        hits += top in c["expected_fault"].split("|")
print(f"INFO  faulty cases: top-1 correct {hits}/{total}")
print(f"INFO  healthy cases: {false_alarms}/{healthy} wrongly flagged at confidence >= 0.5")
assert false_alarms == 0, "healthy machines are being flagged"
ok("rules sanity check finished (accuracy numbers above are for your report notes)")
print("\nALL CHECKS PASSED")