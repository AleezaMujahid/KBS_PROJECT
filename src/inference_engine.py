"""Inference engine (forward + backward chaining, certainty factors)."""

import json

# conflict handling 
COMPETE_MIN = 0.5     
COMPETE_GAP = 0.15     
COMPETE_PENALTY = 0.9  


def combine(cf1, cf2):
    return cf1 + cf2 * (1 - cf1)


class Engine:
    def __init__(self, kb, rules):
        self.kb = kb
        self.rules = rules
        self.symptoms = {s["id"]: s for s in kb["symptoms"]}
        self.faults = {f["id"]: f for f in kb["faults"]}

    @classmethod
    def from_files(cls, kb_path="data/knowledge_base.json",
                   rules_path="data/rules.json"):
        with open(kb_path) as f:
            kb = json.load(f)
        with open(rules_path) as f:
            rules = json.load(f)
        return cls(kb, rules)

    # forward
    def forward(self, facts):
        fired = []
        for r in self.rules:
            if all(facts.get(s, 0) > 0 for s in r["if"]):  # certainty 0 = not present
                evidence = min(facts[s] for s in r["if"])
                fired.append({
                    "rule_id": r["id"], "fault": r["then"], "if": list(r["if"]),
                    "rule_cf": r["cf"], "evidence": evidence,
                    "cf": r["cf"] * evidence, "source": r.get("source", ""),
                })
        return fired

    # perform complete fault diagnosis
    def diagnose(self, facts, unknown=()):
        unknown = set(unknown) - set(facts)
        fired = self.forward(facts)

        by_fault = {}
        for f in fired:
            by_fault.setdefault(f["fault"], []).append(f)

        results = []
        for fault, fs in by_fault.items():
            cf, steps = 0.0, []
            for f in sorted(fs, key=lambda x: -x["cf"]):
                new = combine(cf, f["cf"])
                steps.append({"rule_id": f["rule_id"], "added": f["cf"],
                              "before": cf, "after": new})
                cf = new
            info = self.faults.get(fault, {})
            results.append({
                "fault": fault, "label": info.get("label", fault),
                "cf": cf, "cf_before_conflict": cf, "rules": fs,
                "steps": steps, "competing_with": None,
                "safety_critical": info.get("safety_critical", False),
            })
        results.sort(key=lambda r: -r["cf"])

        competing = self._apply_conflict(results)
        return {"results": results, "facts": dict(facts),
                "questions": self.questions(facts, unknown),
                "competing": competing}

    def _apply_conflict(self, results):
        strong = [r for r in results if r["cf"] >= COMPETE_MIN]
        if len(strong) >= 2 and strong[0]["cf"] - strong[1]["cf"] < COMPETE_GAP:
            a, b = strong[0], strong[1]
            a["competing_with"], b["competing_with"] = b["fault"], a["fault"]
            a["cf"] *= COMPETE_PENALTY
            b["cf"] *= COMPETE_PENALTY
            results.sort(key=lambda r: -r["cf"])
            return True
        return False

    # backward
    def backward(self, hypothesis, facts):
        out = []
        for r in self.rules:
            if r["then"] != hypothesis:
                continue
            have = [s for s in r["if"] if s in facts]
            need = [s for s in r["if"] if s not in facts]
            out.append({"rule_id": r["id"], "rule_cf": r["cf"], "have": have,
                        "need": need, "would_fire": not need})
        out.sort(key=lambda x: (len(x["need"]), -x["rule_cf"]))
        return out

    # follow-up questions
    def questions(self, facts, unknown):
        qs = {}
        for r in self.rules:
            missing = [s for s in r["if"] if s not in facts]
            if missing and all(s in unknown for s in missing):
                for s in missing:
                    q = qs.setdefault(s, {
                        "symptom": s,
                        "label": self.symptoms.get(s, {}).get("label", s),
                        "supports": [], "rules": []})
                    if r["then"] not in q["supports"]:
                        q["supports"].append(r["then"])
                    q["rules"].append(r["id"])
        return list(qs.values())


def apply_answer(facts, unknown, symptom, present):
    facts, unknown = dict(facts), set(unknown)
    unknown.discard(symptom)
    if present:
        facts[symptom] = 1.0
    return facts, unknown
