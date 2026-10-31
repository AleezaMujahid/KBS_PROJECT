"""Helper functions for the Streamlit interface."""

import json
import pandas as pd

from src.preprocess import THRESHOLDS, add_derived
from src.pipeline import reading_to_facts
from src.inference_engine import apply_answer
from src.ranking import recommend
from src.explain import graph_edges

STAKEHOLDERS = {
    "safety_officer": "Safety officer",
    "maintenance_engineer": "Maintenance engineer",
    "plant_manager": "Plant manager",
}

# key: (label, unit, min, max, default, step)  -> input fields in the app
SENSORS = {
    "air": ("Air temperature", "K", 280.0, 330.0, 300.0, 0.1),
    "proc": ("Process temperature", "K", 280.0, 340.0, 310.0, 0.1),
    "rpm": ("Rotational speed", "rpm", 0, 4000, 1500, 1),
    "torque": ("Torque", "Nm", 0.0, 100.0, 40.0, 0.1),
    "wear": ("Tool wear", "min", 0, 400, 100, 1),
}


def load_demo_cases(path="data/demo_cases.json"):
    with open(path) as f:
        return json.load(f)

# process machine readings and follow-up answers to generate a diagnosis and recommendations
def run_with_answers(reading, answers, engine, stakeholder):
    base_facts, base_unknown, missing = reading_to_facts(reading)
    facts, unknown = dict(base_facts), set(base_unknown)
    for sym, present in answers.items():
        facts, unknown = apply_answer(facts, unknown, sym, present)
    diag = engine.diagnose(facts, unknown)
    return {"base_facts": base_facts, "base_unknown": base_unknown,
            "missing": missing, "facts": facts, "unknown": unknown,
            "diagnosis": diag,
            "recommendations": recommend(diag, engine.kb, stakeholder)}


def _fmt(x, fmt):
    return "missing" if pd.isna(x) else fmt.format(x)


def symptom_table(reading, base_facts, base_unknown, answers, engine):
    d = add_derived(pd.DataFrame([{k: (float("nan") if v is None else v)
                                   for k, v in reading.items()}])).iloc[0]
    T = THRESHOLDS
    limit = T["strain_limit"].get(reading["type"])

    def present(sym):
        if sym in base_facts:
            return "yes"
        if sym in base_unknown:
            a = answers.get(sym)
            return "unknown" if a is None else ("yes (answered)" if a else "no (answered)")
        return "no"

    spec = [
        ("Temperature difference (process - air)", _fmt(d["dT"], "{:.1f}") + " K",
         f"< {T['low_temp_diff_K']} K", "low_temp_diff"),
        ("Rotational speed", _fmt(d["rpm"], "{:.0f}") + " rpm",
         f"< {T['low_speed_rpm']} rpm", "low_speed"),
        ("Power (torque x angular speed)", _fmt(d["power"], "{:.0f}") + " W",
         f"< {T['low_power_W']} W", "low_power"),
        ("Power (torque x angular speed)", _fmt(d["power"], "{:.0f}") + " W",
         f"> {T['high_power_W']} W", "high_power"),
        ("Strain (tool wear x torque)", _fmt(d["strain"], "{:.0f}") + " min.Nm",
         f"> {limit} (type {reading['type']})", "high_strain"),
        ("Tool wear", _fmt(d["wear"], "{:.0f}") + " min",
         f">= {T['high_wear_min']} min", "high_tool_wear"),
    ]
    return [{"Derived value": a, "Computed": b, "Symptom if": c,
             "Symptom": engine.symptoms[s]["label"], "Present?": present(s)}
            for a, b, c, s in spec]


def action_rows(recs):
    return [{"#": i, "Action": r["label"], "Priority": round(r["priority"], 3),
             "Safety benefit (1-5)": r["safety_benefit"],
             "Because of": ", ".join(f"{f['fault']} ({f['cf']:.2f})" for f in r["faults"]),
             "Override": "!" if r["override"] else ""}
            for i, r in enumerate(recs, 1)]


def build_graph_html(diag, engine):
    """pyvis graph: symptom -> rule -> fault. Returns HTML (works offline)."""
    from pyvis.network import Network
    net = Network(height="480px", width="100%", directed=True, bgcolor="#ffffff",
                  font_color="#222222", cdn_resources="in_line")
    for sym, rule, fault in graph_edges(diag):
        res = next(x for x in diag["results"] if x["fault"] == fault)
        net.add_node(sym, label=sym, color="#4C9BE8", shape="box",
                     title=engine.symptoms[sym]["label"])
        net.add_node(rule, label=rule, color="#F2B84B", shape="diamond", title="rule")
        net.add_node(fault, label=f"{fault}\n{res['cf']:.2f}", color="#E8554C",
                     shape="ellipse", title=res["label"], size=20 + 30 * res["cf"])
        net.add_edge(sym, rule)
        net.add_edge(rule, fault)
    return net.generate_html(notebook=False)
