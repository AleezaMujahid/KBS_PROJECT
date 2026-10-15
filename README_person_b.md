# Person B - Reasoning Engineer (KBS CT-359)

## Files
- `src/inference_engine.py` - forward chaining, certainty factors, conflict handling, backward chaining, follow-up questions
- `src/ranking.py` - action ranking, 3 stakeholder presets, safety override
- `src/explain.py` - plain-language explanation, graph edges for the pyvis view
- `src/pipeline.py` - raw sensor values -> everything the UI needs (`diagnose_reading`)
- `scripts/evaluate.py` - top-1 / top-3 accuracy on `data/test_cases.csv`
- `tests/test_person_b.py` - 11 checks
- `Report_Sections_4_and_5.docx` - report sections 4 (method comparison) and 5 (implementation and results)
- `evaluation_output.txt`, `evaluation_results.csv` - latest evaluation run

## How to use (copy into the shared repo)
1. Copy `src/`, `scripts/` and `tests/` into the project root, next to Person A's `data/` and `src/preprocess.py`.
2. Run everything from the project root:
   - `python tests/test_person_b.py`
   - `python scripts/evaluate.py`
3. Person C calls: `from src.pipeline import diagnose_reading` and
   `from src.inference_engine import apply_answer`.

Needs Person A's `src/preprocess.py` and `data/` (knowledge_base.json, rules.json, test_cases.csv, demo_cases.json).
Needs `pandas` and `numpy`.

## Data formats expected from Person A
- rules.json: `id`, `if` (list), `then`, `cf`, `source`
- knowledge_base.json actions: `cost`, `downtime_hrs`, `safety_benefit`, `production_impact`; faults: `actions`, `safety_critical`
- test_cases.csv: symptoms separated by `;`, several expected faults separated by `|`, `none` for healthy
