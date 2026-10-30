"""Diagnosis and recommendation explanation."""

def explain_diagnosis(diag, engine):
    lines = []
    facts = diag["facts"]
    if facts:
        lines.append("Symptoms observed:")
        for s, c in facts.items():
            lines.append(f"  - {engine.symptoms[s]['label']}  (certainty {c:.2f})")
    else:
        lines.append("No symptom was observed.")
    if not diag["results"]:
        lines.append("\nNo rule fired: no fault is indicated.")
    for res in diag["results"]:
        lines.append(f"\n{res['label']} ({res['fault']}) - confidence {res['cf']:.2f}")
        for r in res["rules"]:
            lines.append(f"  Rule {r['rule_id']} fired: IF {' AND '.join(r['if'])}")
            lines.append(f"    rule confidence {r['rule_cf']:.2f} x weakest evidence "
                         f"{r['evidence']:.2f} = {r['cf']:.2f}")
            lines.append(f"    source: {r['source']}")
        if len(res["steps"]) > 1:
            lines.append("  Combining rules (cf1 + cf2 x (1 - cf1)):")
            for st in res["steps"][1:]:
                lines.append(f"    {st['before']:.3f} + {st['added']:.3f} x "
                             f"(1 - {st['before']:.3f}) = {st['after']:.3f}")
        if res["competing_with"]:
            lines.append(f"  Competes with {res['competing_with']}: confidence "
                         f"lowered from {res['cf_before_conflict']:.2f} to {res['cf']:.2f}.")
    for q in diag["questions"]:
        lines.append(f"\nQuestion: {q['label']}?  (would support: {', '.join(q['supports'])})")
    return "\n".join(lines)


def explain_recommendations(recs, stakeholder):
    if not recs:
        return "No maintenance action needed."
    lines = [f"Recommended actions for the {stakeholder.replace('_', ' ')}:"]
    for i, r in enumerate(recs, 1):
        why = ", ".join(f"{f['fault']} ({f['cf']:.2f})" for f in r["faults"])
        tag = "  [SAFETY OVERRIDE]" if r["override"] else ""
        lines.append(f"  {i}. {r['label']}  - because of {why}{tag}")
    return "\n".join(lines)

# return symptom, rule, and fault connections for the graph
def graph_edges(diag):
    edges = []
    for res in diag["results"]:
        for r in res["rules"]:
            for s in r["if"]:
                edges.append((s, r["rule_id"], res["fault"]))
    return edges
