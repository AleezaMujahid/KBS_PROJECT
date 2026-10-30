"""Rank maintenance actions for a stakeholder."""

PRESETS = {   
    "safety_officer":       {"safety": 0.60, "downtime": 0.10, "cost": 0.10, "production": 0.20},
    "maintenance_engineer": {"safety": 0.25, "downtime": 0.30, "cost": 0.30, "production": 0.15},
    "plant_manager":        {"safety": 0.20, "downtime": 0.20, "cost": 0.20, "production": 0.40},
}
MIN_CF = 0.05            # ignore faults weaker than this
OVERRIDE_CF = 0.5        # a safety-critical fault this sure triggers the override


def action_score(action, weights, max_downtime):
    safety = (action["safety_benefit"] - 1) / 4
    cost = 1 - (action["cost"] - 1) / 4
    prod = 1 - (action["production_impact"] - 1) / 4
    down = 1 - (action["downtime_hrs"] / max_downtime if max_downtime else 0)
    return (weights["safety"] * safety + weights["downtime"] * down
            + weights["cost"] * cost + weights["production"] * prod)


# recommended actions for stakeholders
def recommend(diagnosis, kb, stakeholder="safety_officer"):
    weights = PRESETS[stakeholder]
    actions = {a["id"]: a for a in kb["actions"]}
    faults = {f["id"]: f for f in kb["faults"]}
    max_down = max((a["downtime_hrs"] for a in actions.values()), default=0)

    found = {}
    for res in diagnosis["results"]:
        if res["cf"] < MIN_CF:
            continue
        critical = (faults[res["fault"]].get("safety_critical", False)
                    and res["cf"] >= OVERRIDE_CF)
        for aid in faults[res["fault"]]["actions"]:
            a = actions[aid]
            s = action_score(a, weights, max_down)
            item = found.setdefault(aid, {
                "id": aid, "label": a["label"], "score": s, "priority": 0.0,
                "safety_benefit": a["safety_benefit"], "faults": [],
                "override": False})
            item["priority"] = max(item["priority"], s * res["cf"])
            item["faults"].append({"fault": res["fault"], "cf": res["cf"]})
            item["override"] = item["override"] or critical

    ranked = list(found.values())
    ranked.sort(key=lambda x: (not x["override"],
                               -x["safety_benefit"] if x["override"] else 0,
                               -x["priority"]))
    return ranked
