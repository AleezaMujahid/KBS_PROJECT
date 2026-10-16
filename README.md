# Industrial Fault Diagnosis: Knowledge-Based System (CT-359)

A rule-based expert system (not a trained ML model) that diagnoses faults in an
industrial machine tool (milling / CNC type) from sensor readings and recommends
maintenance actions ranked for different stakeholders.

Data basis: the UCI **AI4I 2020 Predictive Maintenance** dataset (10,000 records,
5 failure modes). Note: AI4I is *synthetic*, built to resemble real industrial data.

## Quick start

```bash
python -m venv venv
venv\Scripts\activate          # Windows   (Mac/Linux: source venv/bin/activate)
pip install -r requirements.txt
streamlit run app.py           # run from the project root
```

## What the app does

| Step | What you see |
|------|--------------|
| 1. Sensor readings | Product type, air/process temperature, speed, torque, tool wear. Untick **sensor working** to simulate a missing sensor. |
| 2. Sensor -> symptoms | Table showing derived values (temperature difference, power, strain), the threshold used and whether each symptom is present. |
| 3. Follow-up questions | For symptoms the engine could not observe, it asks Yes/No. Each answer re-runs the diagnosis. |
| 4. Results | Tabs: **Diagnosis** (ranked faults with confidence), **Recommended actions** (ranked per stakeholder, safety override), **Explanation** (which rules fired and why), **Reasoning graph** (pyvis: symptom -> rule -> fault), **What-if** (backward chaining: evidence still needed for a chosen fault). |

Sidebar: choose the **stakeholder** (safety officer / maintenance engineer / plant manager)
and **load a sample machine reading** (5 demo cases: healthy, overheating, power problem,
two faults at once, missing air-temperature sensor).

### Suggested demo order
1. *Healthy machine*: no fault.
2. *Overheating*: HDF with the full explanation and graph.
3. Switch stakeholder on the same case and show the action ranking change.
4. *Missing information*: answer the follow-up question and watch confidence update.
5. *Two faults at once*: competing faults and the What-if tab.

## Project structure

```
app.py                  Streamlit interface (Person C)
src/ui_helpers.py       logic behind the app: tables, graph, re-run with answers (Person C)
src/preprocess.py       sensor values -> symptoms, all thresholds in one table (Person A)
src/inference_engine.py forward + backward chaining, certainty factors (Person B)
src/ranking.py          stakeholder presets, safety override (Person B)
src/explain.py          explanation text, graph edges (Person B)
src/pipeline.py         raw reading -> facts/unknowns (Person B)
data/                   knowledge_base.json, rules.json, test_cases.csv,
                        demo_cases.json, ai4i2020.csv
scripts/evaluate.py     top-1 / top-3 accuracy on the test cases
tests/                  checks for Person A, B and C code
```

## Tests and evaluation

```bash
python tests/test_person_a.py
python tests/test_person_b.py
python tests/test_person_c.py
python scripts/evaluate.py
```

## How the reasoning works (short)

- **Forward chaining**: a rule fires when all its IF symptoms are present.
  Rule strength = rule confidence x weakest matched evidence.
- Two rules for the same fault combine as `cf1 + cf2 x (1 - cf1)`.
- If two strong faults are close, both confidences are lowered (conflict).
- **Backward chaining**: for a hypothesised fault, list which evidence is still needed.
- **Ranking**: weighted score over safety, downtime, cost and production impact;
  a safety-critical fault above the confidence limit triggers a safety override.
