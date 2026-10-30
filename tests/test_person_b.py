"""Tests for inference, diagnosis, and recommendation functionality.
Run from the project root using: python tests/test_person_b.py"""

import json
import sys
sys.path.insert(0, ".")
from src.inference_engine import Engine, combine, apply_answer
from src.ranking import recommend, PRESETS
from src.pipeline import diagnose_reading
from src.explain import explain_diagnosis, graph_edges

ok = lambda m: print("PASS ", m)
E = Engine.from_files()

# test certainty factor combination
assert abs(combine(0.95, 0.16) - 0.958) < 1e-9
assert abs(combine(0.6, 0.5) - 0.8) < 1e-9
ok("combine: 0.95 and 0.16 -> 0.958 ; 0.6 and 0.5 -> 0.8")

# test forward chaining and evidence strength
d = E.diagnose({"low_temp_diff": 1.0, "low_speed": 0.8})
top = d["results"][0]
assert top["fault"] == "HDF" and len(top["rules"]) == 2
assert any(abs(r["cf"] - 0.95 * 0.8) < 1e-9 for r in top["rules"])
ok("forward chaining fires R1 + R2; rule cf = rule cf x weakest evidence")

# test diagnosis with no symptoms
assert E.diagnose({})["results"] == []
ok("no symptoms -> no fault")

# test backward chaining and missing evidence
b = E.backward("HDF", {"low_speed": 1.0})
r1 = [x for x in b if x["rule_id"] == "R1"][0]
assert r1["need"] == ["low_temp_diff"] and r1["have"] == ["low_speed"]
ok("backward chaining: HDF still needs low_temp_diff")

# test conflict handling
d = E.diagnose({"low_power": 1.0, "high_strain": 1.0})
assert d["competing"] and all(r["cf"] < 0.95 for r in d["results"][:2])
ok("two strong faults close together -> flagged and lowered")

# test follow-up questions for missing sensor data
demo = json.load(open("data/demo_cases.json"))
miss = [c for c in demo if c["name"] == "Missing information"][0]
reading = {k: miss[k] for k in ("type", "air", "proc", "rpm", "torque", "wear")}
out = diagnose_reading(reading, engine=E)
assert out["missing_sensors"] == ["air"]
assert any(q["symptom"] == "low_temp_diff" for q in out["diagnosis"]["questions"])
facts, unknown = apply_answer(out["facts"], out["unknown"], "low_temp_diff", True)
d2 = E.diagnose(facts, unknown)
assert {r["fault"] for r in d2["results"]} >= {"OSF", "HDF"}
ok("missing air temp -> engine asks; answering Yes re-runs diagnosis")

# test safety override across stakeholder presets
d = E.diagnose({"high_strain": 1.0})
for p in PRESETS:
    recs = recommend(d, E.kb, p)
    assert recs and recs[0]["override"], p
ok("safety override works for all 3 stakeholder presets")

# test recommendations for low-confidence faults
d = E.diagnose({"high_tool_wear": 1.0})
for p in PRESETS:
    recs = recommend(d, E.kb, p)
    assert recs[0]["id"] == "replace_tool" and not recs[0]["override"], p
ok("low-confidence TWF still gives a (low priority) action, no override")

# test zero-certainty symptoms
assert E.diagnose({"low_power": 0.0})["results"] == []
ok("certainty 0 is treated as absent")

# test diagnosis explanations and graph generation
assert "Rule R1" in explain_diagnosis(E.diagnose({"low_temp_diff": 1.0, "low_speed": 1.0}), E)
assert ("low_speed", "R1", "HDF") in graph_edges(E.diagnose({"low_temp_diff": 1.0, "low_speed": 1.0}))
ok("explain.py + graph_edges work")

# test demo cases through the complete pipeline
for c in demo:
    r = {k: c[k] for k in ("type", "air", "proc", "rpm", "torque", "wear")}
    diagnose_reading(r, engine=E)
ok("5 demo cases run through the whole pipeline")
print("\nALL CHECKS PASSED")
