"""Person B's checks. Run from the project root:  python tests/test_person_b.py"""
import json
import sys
sys.path.insert(0, ".")
from src.inference_engine import Engine, combine, apply_answer
from src.ranking import recommend, PRESETS
from src.pipeline import diagnose_reading
from src.explain import explain_diagnosis, graph_edges

ok = lambda m: print("PASS ", m)
E = Engine.from_files()

# 1. the combine formula
assert abs(combine(0.95, 0.16) - 0.958) < 1e-9
assert abs(combine(0.6, 0.5) - 0.8) < 1e-9
ok("combine: 0.95 and 0.16 -> 0.958 ; 0.6 and 0.5 -> 0.8")

# 2. forward chaining + weakest evidence
d = E.diagnose({"low_temp_diff": 1.0, "low_speed": 0.8})
top = d["results"][0]
assert top["fault"] == "HDF" and len(top["rules"]) == 2
assert any(abs(r["cf"] - 0.95 * 0.8) < 1e-9 for r in top["rules"])
ok("forward chaining fires R1 + R2; rule cf = rule cf x weakest evidence")

# 3. nothing observed -> nothing fires
assert E.diagnose({})["results"] == []
ok("no symptoms -> no fault")

# 4. backward chaining lists the missing evidence
b = E.backward("HDF", {"low_speed": 1.0})
r1 = [x for x in b if x["rule_id"] == "R1"][0]
assert r1["need"] == ["low_temp_diff"] and r1["have"] == ["low_speed"]
ok("backward chaining: HDF still needs low_temp_diff")

# 5. conflict lowers confidence
d = E.diagnose({"low_power": 1.0, "high_strain": 1.0})
assert d["competing"] and all(r["cf"] < 0.95 for r in d["results"][:2])
ok("two strong faults close together -> flagged and lowered")

# 6. follow-up question flow (missing air temperature)
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

# 7. safety override puts a safety action first, whatever the preset
d = E.diagnose({"high_strain": 1.0})
for p in PRESETS:
    recs = recommend(d, E.kb, p)
    assert recs and recs[0]["override"], p
ok("safety override works for all 3 stakeholder presets")

# 8. low-confidence, non-critical fault still gives an action, with no override
d = E.diagnose({"high_tool_wear": 1.0})
for p in PRESETS:
    recs = recommend(d, E.kb, p)
    assert recs[0]["id"] == "replace_tool" and not recs[0]["override"], p
ok("low-confidence TWF still gives a (low priority) action, no override")

# 8b. a symptom with certainty 0 must not fire a rule
assert E.diagnose({"low_power": 0.0})["results"] == []
ok("certainty 0 is treated as absent")

# 9. explanation and graph edges are produced
assert "Rule R1" in explain_diagnosis(E.diagnose({"low_temp_diff": 1.0, "low_speed": 1.0}), E)
assert ("low_speed", "R1", "HDF") in graph_edges(E.diagnose({"low_temp_diff": 1.0, "low_speed": 1.0}))
ok("explain.py + graph_edges work")

# 10. all five demo cases run end to end
for c in demo:
    r = {k: c[k] for k in ("type", "air", "proc", "rpm", "torque", "wear")}
    diagnose_reading(r, engine=E)
ok("5 demo cases run through the whole pipeline")
print("\nALL CHECKS PASSED")
