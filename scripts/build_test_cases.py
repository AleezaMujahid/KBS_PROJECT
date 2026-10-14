"""Build data/test_cases.csv and data/demo_cases.json from the AI4I data.
Run from the project root:  python scripts/build_test_cases.py
Expected faults come from the LABEL columns, never from our rules."""
import json
import sys
sys.path.insert(0, ".")
import pandas as pd
from src.preprocess import load_data, row_to_symptoms

SEED = 42
MODES = ["TWF", "HDF", "PWF", "OSF"]

df = load_data("data/ai4i2020.csv")
n_modes = df[MODES + ["RNF"]].sum(axis=1)

parts = []
for m in MODES:                                   # 10 clean cases per fault
    pool = df[(n_modes == 1) & (df[m] == 1)]
    parts.append(pool.sample(10, random_state=SEED))
multi = df[(n_modes > 1) & (df["RNF"] == 0)]      # 5 cases with two faults at once
parts.append(multi.sample(5, random_state=SEED))
healthy = df[df["fail"] == 0]                     # 15 healthy machines
parts.append(healthy.sample(15, random_state=SEED))
cases = pd.concat(parts)


def expected(row):
    faults = [m for m in MODES if row[m] == 1]
    return "|".join(faults) if faults else "none"


out = pd.DataFrame({
    "case_id": [f"T{i+1:02d}" for i in range(len(cases))],
    "symptoms": [";".join(sorted(row_to_symptoms(r))) for _, r in cases.iterrows()],
    "expected_fault": [expected(r) for _, r in cases.iterrows()],
    "type": cases["type"].values, "air": cases["air"].values,
    "proc": cases["proc"].values, "rpm": cases["rpm"].values,
    "torque": cases["torque"].values, "wear": cases["wear"].values,
})
out.to_csv("data/test_cases.csv", index=False)
print("test_cases.csv:", len(out), "rows")
print(out["expected_fault"].value_counts().to_string())

# ---- 5 demo cases for Person C (raw sensor values) ----
def pick(mask, label, note, seed=SEED):
    r = df[mask].sample(1, random_state=seed).iloc[0]
    return {"name": label, "note": note, "type": r["type"], "air": r["air"],
            "proc": r["proc"], "rpm": int(r["rpm"]), "torque": r["torque"],
            "wear": int(r["wear"]), "label": expected(r)}

demo = [
    pick(df["fail"] == 0, "Healthy machine", "Should produce no fault"),
    pick((n_modes == 1) & (df["HDF"] == 1), "Overheating", "Heat dissipation failure"),
    pick((n_modes == 1) & (df["PWF"] == 1), "Power problem", "Power failure"),
    pick((n_modes > 1) & (df["RNF"] == 0), "Two faults at once", "Two faults compete"),
]
d5 = pick((n_modes == 1) & (df["OSF"] == 1), "Missing information",
          "Air temperature sensor is missing: engine should ask for it")
d5["air"] = None
demo.append(d5)
with open("data/demo_cases.json", "w") as f:
    json.dump(demo, f, indent=2)
print("demo_cases.json:", len(demo), "cases")