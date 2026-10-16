"""Person C's checks. Run from the project root:  python tests/test_person_c.py"""
import os
import sys
sys.path.insert(0, ".")
from src.inference_engine import Engine
from src.ui_helpers import (STAKEHOLDERS, SENSORS, load_demo_cases,
                            run_with_answers, symptom_table, action_rows,
                            build_graph_html)

ok = lambda m: print("PASS ", m)
E = Engine.from_files()
CASES = {c["name"]: c for c in load_demo_cases()}


def reading(case):
    return {k: case[k] for k in ["type", "air", "proc", "rpm", "torque", "wear"]}


# 1. every demo case runs through the app logic for every stakeholder
for case in CASES.values():
    for sh in STAKEHOLDERS:
        out = run_with_answers(reading(case), {}, E, sh)
        assert "diagnosis" in out and "recommendations" in out
ok("5 demo cases x 3 stakeholders run through run_with_answers")

# 2. top fault matches the expected label (healthy -> nothing)
out = run_with_answers(reading(CASES["Healthy machine"]), {}, E, "safety_officer")
assert out["diagnosis"]["results"] == []
out = run_with_answers(reading(CASES["Overheating"]), {}, E, "safety_officer")
assert out["diagnosis"]["results"][0]["fault"] == "HDF"
out = run_with_answers(reading(CASES["Power problem"]), {}, E, "safety_officer")
assert out["diagnosis"]["results"][0]["fault"] == "PWF"
ok("healthy -> none ; overheating -> HDF ; power problem -> PWF")

# 3. missing sensor -> follow-up question, and the answer changes the result
r = reading(CASES["Missing information"])
before = run_with_answers(r, {}, E, "safety_officer")
qs = before["diagnosis"]["questions"]
assert [q["symptom"] for q in qs] == ["low_temp_diff"]
assert "air" in before["missing"]
after = run_with_answers(r, {"low_temp_diff": True}, E, "safety_officer")
assert after["diagnosis"]["questions"] == []
assert after["diagnosis"]["results"][0]["cf"] != before["diagnosis"]["results"][0]["cf"]
no = run_with_answers(r, {"low_temp_diff": False}, E, "safety_officer")
assert no["diagnosis"]["questions"] == []
ok("follow-up question appears, and Yes / No both re-run the diagnosis")

# 4. the symptom table covers all 6 symptoms and shows 'missing'
rows = symptom_table(r, before["base_facts"], before["base_unknown"], {}, E)
assert len(rows) == 6
assert any(x["Computed"].startswith("missing") for x in rows)
assert any(x["Present?"] == "unknown" for x in rows)
rows2 = symptom_table(r, after["base_facts"], after["base_unknown"],
                      {"low_temp_diff": True}, E)
assert any(x["Present?"] == "yes (answered)" for x in rows2)
ok("symptom table: 6 rows, shows missing / unknown / answered")

# 5. stakeholder changes the action ranking numbers
a = run_with_answers(reading(CASES["Overheating"]), {}, E, "safety_officer")
b = run_with_answers(reading(CASES["Overheating"]), {}, E, "plant_manager")
assert [x["priority"] for x in a["recommendations"]] != \
       [x["priority"] for x in b["recommendations"]]
assert len(action_rows(a["recommendations"])) == len(a["recommendations"])
ok("different stakeholder -> different action priorities")

# 6. pyvis graph builds offline (no CDN link) and contains the fault
html = build_graph_html(a["diagnosis"], E)
assert "HDF" in html and ("vis-network" in html.lower() or "vis.Network" in html)
assert "cdnjs" not in html and "unpkg" not in html
ok("pyvis graph HTML is self-contained")

# 7. the Streamlit app itself runs, with no exception, on every demo case
try:
    from streamlit.testing.v1 import AppTest
except ImportError:
    print("SKIP  streamlit not installed - app test skipped")
else:
    at = AppTest.from_file(os.path.abspath("app.py"), default_timeout=60).run()
    assert not at.exception
    for i in range(len(CASES)):
        at.session_state["sample_idx"] = i
        [x for x in at.button if x.label == "Load sample reading"][0].click().run()
        [x for x in at.button if "Diagnose" in x.label][0].click().run()
        assert not at.exception, at.exception
    ok("app.py runs on all demo cases without errors")

print("\nALL CHECKS PASSED")
