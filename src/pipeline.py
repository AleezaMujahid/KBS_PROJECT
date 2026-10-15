"""One call from raw sensor values to everything Person C needs.  Owner: Person B."""
import numpy as np
import pandas as pd
from src.preprocess import add_derived, row_to_symptoms
from src.inference_engine import Engine
from src.ranking import recommend
from src.explain import explain_diagnosis, explain_recommendations

# if a sensor is missing, these symptoms become 'unknown' (we can ask about them)
SENSOR_SYMPTOMS = {
    "air": ["low_temp_diff"], "proc": ["low_temp_diff"],
    "rpm": ["low_speed", "low_power", "high_power"],
    "torque": ["low_power", "high_power", "high_strain"],
    "wear": ["high_tool_wear", "high_strain"],
    "type": ["high_strain"],
}


def reading_to_facts(reading):
    """reading = dict(type, air, proc, rpm, torque, wear); None = sensor missing.
    Returns facts, unknown symptoms, missing sensor names."""
    r = {k: (np.nan if v is None else v) for k, v in reading.items()}
    missing = [k for k in SENSOR_SYMPTOMS if pd.isna(r.get(k, np.nan))]
    row = add_derived(pd.DataFrame([r])).iloc[0]
    facts = row_to_symptoms(row)
    unknown = {s for m in missing for s in SENSOR_SYMPTOMS[m]} - set(facts)
    return facts, unknown, missing


def diagnose_reading(reading, stakeholder="safety_officer", engine=None):
    engine = engine or Engine.from_files()
    facts, unknown, missing = reading_to_facts(reading)
    diag = engine.diagnose(facts, unknown)
    recs = recommend(diag, engine.kb, stakeholder)
    return {"facts": facts, "unknown": unknown, "missing_sensors": missing,
            "diagnosis": diag, "recommendations": recs,
            "explanation": explain_diagnosis(diag, engine),
            "action_text": explain_recommendations(recs, stakeholder)}
