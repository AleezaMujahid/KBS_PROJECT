"""Streamlit app: Industrial Fault Diagnosis.

Run from the project root:   streamlit run app.py

Flow:  sensor values -> symptoms 
       -> inference engine -> ranked faults, actions, explanation
       -> follow-up questions (Yes/No) re-run the diagnosis
"""
import os
import sys
from pathlib import Path

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

ROOT = Path(__file__).resolve().parent
os.chdir(ROOT)                      # engine uses relative data/ paths
sys.path.insert(0, str(ROOT))

from src.inference_engine import Engine                     # noqa: E402
from src.ranking import PRESETS                             # noqa: E402
from src.explain import (explain_diagnosis, explain_recommendations,  # noqa: E402
                         graph_edges)
from src.ui_helpers import (STAKEHOLDERS, SENSORS, load_demo_cases,  # noqa: E402
                            run_with_answers, symptom_table, action_rows,
                            build_graph_html)

st.set_page_config(page_title="Industrial Fault Diagnosis KBS",
                   page_icon="🛠️", layout="wide")



# ------------------------------------------- Streamlit version compatibility
def stretch():
    """width kwarg differs between old and new Streamlit versions."""
    import inspect
    if "width" in inspect.signature(st.button).parameters:
        return {"width": "stretch"}
    return {"use_container_width": True}


STRETCH = stretch()


def show_html(html, height):
    if hasattr(st, "iframe"):
        try:
            st.iframe(html, height=height)
            return
        except Exception:
            pass
    components.html(html, height=height, scrolling=False)


# --------------------------------------------------------------- cached data
@st.cache_resource
def get_engine():
    return Engine.from_files()


@st.cache_data
def get_demo_cases():
    return load_demo_cases()


engine = get_engine()
demo_cases = get_demo_cases()


# ------------------------------------------------------------- state helpers
def init_state():
    st.session_state.setdefault("type", "M")
    for k, (_, _, _, _, default, _) in SENSORS.items():
        st.session_state.setdefault(k, default)
        st.session_state.setdefault(f"has_{k}", True)
    st.session_state.setdefault("reading", None)       # set by Diagnose button
    st.session_state.setdefault("answers", {})         # follow-up answers
    st.session_state.setdefault("case_note", "")


def load_sample():
    case = demo_cases[st.session_state["sample_idx"]]
    st.session_state["type"] = case["type"]
    for k in SENSORS:
        v = case.get(k)
        st.session_state[f"has_{k}"] = v is not None
        if v is not None:
            st.session_state[k] = type(SENSORS[k][4])(v)
    st.session_state["case_note"] = f"{case['name']}: {case['note']}"
    st.session_state["reading"] = None
    st.session_state["answers"] = {}


def run_diagnosis():
    reading = {"type": st.session_state["type"]}
    for k in SENSORS:
        reading[k] = st.session_state[k] if st.session_state[f"has_{k}"] else None
    st.session_state["reading"] = reading
    st.session_state["answers"] = {}


def answer(symptom, present):
    st.session_state["answers"][symptom] = present


init_state()

# ------------------------------------------------------------------- header
st.title("🛠️ Industrial Fault Diagnosis")
st.caption("Knowledge-based system (rule-based expert system) for a milling / "
           "CNC machine tool. Data basis: UCI AI4I 2020 (synthetic data "
           "built to resemble real industrial data).")

# ------------------------------------------------------------------ sidebar
with st.sidebar:
    st.header("Who is asking?")
    stakeholder = st.selectbox(
        "Stakeholder", list(STAKEHOLDERS), format_func=STAKEHOLDERS.get,
        key="stakeholder",
        help="Changes how maintenance actions are ranked.")
    w = PRESETS[stakeholder]
    st.caption("Ranking weights: " + ", ".join(f"{k} {v:.0%}" for k, v in w.items()))

    st.divider()
    st.header("Sample machine reading")
    st.selectbox("Choose a sample", range(len(demo_cases)),
                 format_func=lambda i: demo_cases[i]["name"], key="sample_idx")
    st.button("Load sample reading", on_click=load_sample, **STRETCH)
    if st.session_state["case_note"]:
        st.info(st.session_state["case_note"])

# --------------------------------------------------------------- input form
st.subheader("1. Sensor readings")
cols = st.columns([1] + [1] * len(SENSORS))
cols[0].selectbox("Product type", ["L", "M", "H"], key="type",
                  help="Low / Medium / High quality variant")
for col, (k, (label, unit, lo, hi, default, step)) in zip(cols[1:], SENSORS.items()):
    with col:
        st.checkbox("sensor working", key=f"has_{k}")
        st.number_input(f"{label} [{unit}]", min_value=type(default)(lo),
                        max_value=type(default)(hi), step=type(default)(step),
                        key=k, disabled=not st.session_state[f"has_{k}"])

st.button("🔍 Diagnose", type="primary", on_click=run_diagnosis)

reading = st.session_state["reading"]
if reading is None:
    st.info("Enter sensor values (or load a sample) and press **Diagnose**.")
    st.stop()

# ------------------------------------------------- reasoning (src/ui_helpers.py)
out = run_with_answers(reading, st.session_state["answers"], engine, stakeholder)
base_facts, base_unknown, missing = out["base_facts"], out["base_unknown"], out["missing"]
facts, diag, recs = out["facts"], out["diagnosis"], out["recommendations"]

# ------------------------------------------- sensor -> symptom on screen
st.subheader("2. Sensor values → symptoms")
st.dataframe(pd.DataFrame(symptom_table(reading, base_facts, base_unknown,
                                        st.session_state["answers"], engine)),
             hide_index=True, **STRETCH)
if missing:
    st.warning("Missing sensor(s): " + ", ".join(missing) +
               ". Symptoms that depend on them are unknown; see follow-up questions.")

# ----------------------------------------------------- follow-up questions
if diag["questions"]:
    st.subheader("3. Follow-up questions")
    st.caption("The engine could not observe these symptoms. Your answer "
               "re-runs the diagnosis.")
    for q in diag["questions"]:
        c1, c2, c3 = st.columns([5, 1, 1])
        c1.markdown(f"**{q['label']}?**  \n"
                    f"<small>would support: {', '.join(q['supports'])}</small>",
                    unsafe_allow_html=True)
        c2.button("Yes", key=f"yes_{q['symptom']}", on_click=answer,
                  args=(q["symptom"], True), **STRETCH)
        c3.button("No", key=f"no_{q['symptom']}", on_click=answer,
                  args=(q["symptom"], False), **STRETCH)
elif st.session_state["answers"]:
    st.success("All follow-up questions answered.")

# ----------------------------------------------------------- result panels
st.subheader("4. Results")
tab_diag, tab_act, tab_why, tab_graph, tab_back = st.tabs(
    ["Diagnosis", "Recommended actions", "Explanation", "Reasoning graph",
     "What-if (backward chaining)"])

with tab_diag:
    if not diag["results"]:
        st.success("No rule fired: no fault is indicated. Machine looks healthy.")
    else:
        if diag["competing"]:
            st.warning("Two strong faults compete; the engine lowered both "
                       "confidences because it is less sure which is the cause.")
        for r in diag["results"]:
            a, b = st.columns([1, 3])
            crit = " ⚠️ safety-critical" if r["safety_critical"] else ""
            a.markdown(f"**{r['label']}** ({r['fault']}){crit}")
            b.progress(min(max(r["cf"], 0.0), 1.0), text=f"confidence {r['cf']:.2f}")

with tab_act:
    st.caption(f"Ranked for: {STAKEHOLDERS[stakeholder]}")
    if not recs:
        st.success("No maintenance action needed.")
    else:
        if any(r["override"] for r in recs):
            st.error("SAFETY OVERRIDE: a safety-critical fault is likely, so its "
                     "actions come first for every stakeholder.")
        st.dataframe(pd.DataFrame(action_rows(recs)), hide_index=True, **STRETCH)
        st.text(explain_recommendations(recs, stakeholder))

with tab_why:
    st.code(explain_diagnosis(diag, engine), language="text")



with tab_graph:
    if not graph_edges(diag):
        st.info("No rule fired, so there is no reasoning path to draw.")
    else:
        st.caption("Blue = symptoms, yellow = rules that fired, red = faults "
                   "(larger = more confident).")
        show_html(build_graph_html(diag, engine), 520)

with tab_back:
    st.caption("Backward chaining: pick a fault and see which evidence is "
               "still needed to prove it.")
    hyp = st.selectbox("Hypothesis", list(engine.faults),
                       format_func=lambda f: f"{f} - {engine.faults[f]['label']}")
    for item in engine.backward(hyp, facts):
        mark = "✅ would fire" if item["would_fire"] else "❌ not yet"
        st.markdown(
            f"**Rule {item['rule_id']}** (cf {item['rule_cf']:.2f}) {mark}  \n"
            f"have: {', '.join(item['have']) or '-'}  \n"
            f"still needed: {', '.join(item['need']) or '-'}")
