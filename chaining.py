import difflib
import os
import sys
from collections import namedtuple

from atomspace import STV, Atom, AtomSpace, Concept, Link, Predicate
from pln_rules import (abduction_link, deduction_link, induction_link,
                       is_no_info, revision)

PATIENT_BASE_RATE = (0.01, 0.1)           # (strength, confidence) of the patient node
DEFAULT_OBSERVATION_CONFIDENCE = 0.9
MIN_CONFIDENCE = 0.01
MAX_ROUNDS = 3
MAX_DEPTH = 3
SUGGESTION_LIMIT = 5
HAS_SYMPTOM = "has_symptom"

Observation = namedtuple("Observation", "id symptom stv")
Ranked = namedtuple("Ranked", "name strength confidence")
Suggestion = namedtuple("Suggestion", "symptom strength confidence via")
Missing = namedtuple("Missing", "symptom strength confidence disease")

def fmt(stv):
    """Readable truth value: (strength, confidence)."""
    return f"({stv.strength:.4f}, {stv.confidence:.4f})"

def link_text(atom):
    """Readable text for a binary link with its truth value."""
    return f"{atom.targets[0].name} -> {atom.targets[1].name} {fmt(atom.stv)}"

def copy_space(source):
    """Return an independent copy of an atom space."""
    target = AtomSpace()

    def copy_atom(atom):
        stv = STV(atom.stv.strength, atom.stv.confidence)
        if atom.is_node:
            return target.add(Atom(atom.type, name=atom.name, stv=stv))
        targets = [copy_atom(t) for t in atom.targets]
        return target.add(Atom(atom.type, targets=targets, stv=stv))

    for atom in source.all():
        copy_atom(atom)
    return target

def _did_you_mean(name, pool):
    matches = difflib.get_close_matches(name, list(pool), n=3)
    return f" Did you mean: {', '.join(matches)}?" if matches else ""

class Belief:
    """A truth value together with the evidence IDs it rests on."""

    __slots__ = ("stv", "evidence")

    def __init__(self, stv, evidence):
        self.stv = stv
        self.evidence = frozenset(evidence)

CHANGING_ACTIONS = {"new", "revised", "replaced"}

def merge_belief(existing, candidate):
    if existing is None:
        return candidate, "new"
    if candidate.evidence <= existing.evidence:
        return existing, "already counted"
    if existing.evidence.isdisjoint(candidate.evidence):
        merged = Belief(revision(existing.stv, candidate.stv),
                        existing.evidence | candidate.evidence)
        return merged, "revised"
    if ((candidate.stv.confidence, candidate.stv.strength)
            > (existing.stv.confidence, existing.stv.strength)):
        return candidate, "replaced"
    return existing, "kept existing"

class Derivation:
    """One application of a rule: premises, conclusion and evidence."""

    __slots__ = ("rule", "premises", "atom", "evidence", "absent")

    def __init__(self, rule, premises, atom, evidence, absent=False):
        self.rule = rule
        self.premises = premises
        self.atom = atom
        self.evidence = frozenset(evidence)
        self.absent = absent

class TraceEntry:
    """One line of the inference log."""

    def __init__(self, round_no, rule, premises, conclusion, target, stv, note=""):
        self.round = round_no
        self.rule = rule
        self.premises = premises
        self.conclusion = conclusion
        self.target = target
        self.stv = stv
        self.note = note

    def __str__(self):
        left = " | ".join(self.premises)
        right = self.conclusion
        if self.stv is not None:
            right += f" {fmt(self.stv)}"
        text = f"[round {self.round}] {self.rule}: {left}  =>  {right}"
        if self.note:
            text += f"   ({self.note})"
        return text

class ProofNode:
    """A node of the backward chaining proof tree."""

    def __init__(self, goal, rule, stv=None, children=None, note=""):
        self.goal = goal
        self.rule = rule
        self.stv = stv
        self.children = children or []
        self.note = note

    def lines(self, indent=0):
        head = f"{'  ' * indent}{self.goal}  [{self.rule}]"
        if self.stv is not None:
            head += f"  {fmt(self.stv)}"
        if self.note:
            head += f"  - {self.note}"
        out = [head]
        for child in self.children:
            out.extend(child.lines(indent + 1))
        return out

    def __str__(self):
        return "\n".join(self.lines())

class _KBInfo:
    """Diseases, categories and symptoms read from the knowledge base."""

    def __init__(self, kb):
        disease_categories = {}
        category_diseases = {}
        for link in kb.of_type("InheritanceLink"):
            if len(link.targets) != 2:
                continue
            d, c = link.targets
            if not (d.is_node and c.is_node):
                continue
            disease_categories.setdefault(d.name, set()).add(c.name)
            category_diseases.setdefault(c.name, set()).add(d.name)
        if not disease_categories:
            raise ValueError("the knowledge base has no Disease -> Category links")

        self.diseases = sorted(disease_categories)
        self.categories = sorted(category_diseases)
        self.disease_categories = {d: sorted(v) for d, v in disease_categories.items()}
        self.category_diseases = {c: sorted(v) for c, v in category_diseases.items()}

        self.disease_symptoms = {}       # disease -> {symptom name: link}
        self.absent_confidence = {}      # disease -> confidence of a missing link
        symptoms = set()
        for d in self.diseases:
            node = kb.get("ConceptNode", name=d)
            links = {}
            for link in kb.links_from(node, "ImplicationLink"):
                if len(link.targets) == 2 and link.targets[1].is_node:
                    links[link.targets[1].name] = link
            self.disease_symptoms[d] = links
            symptoms.update(links)
            if links:
                self.absent_confidence[d] = max(l.stv.confidence for l in links.values())
        self.symptoms = sorted(symptoms)

        self._disease_set = set(self.diseases)
        self._category_set = set(self.categories)
        self._symptom_set = set(self.symptoms)

    def kind(self, name):
        if name in self._disease_set:
            return "disease"
        if name in self._category_set:
            return "category"
        if name in self._symptom_set:
            return "symptom"
        pool = self.diseases + self.categories + self.symptoms
        raise ValueError(f"unknown name '{name}'." + _did_you_mean(name, pool))

class _Workspace:
    """Everything one chaining run needs. Built fresh for every run."""

    def __init__(self, engine):
        self.info = engine.info
        self.space = copy_space(engine.kb)
        self.patient = self.space.add(
            Concept(engine.patient, STV(*PATIENT_BASE_RATE)))
        self.predicate = self.space.add(Predicate(HAS_SYMPTOM))
        self.trace = []
        self.symptom_beliefs = {}
        self.max_depth = MAX_DEPTH
        self.min_confidence = MIN_CONFIDENCE
        self._merge_reports(engine.observations())

    def node(self, name):
        return self.space.get("ConceptNode", name=name)

    def store(self, kind, name, belief):
        """Write a belief into the workspace as Patient -> name."""
        node = self.node(name)
        self.space.add(Link("ImplicationLink", self.patient, node, stv=belief.stv))
        if kind == "symptom":
            self.space.add(Link("EvaluationLink", self.predicate, self.patient,
                                node, stv=belief.stv))

    def _merge_reports(self, observations):
        """Reports of the same symptom are independent evidence: merge by revision."""
        groups = {}
        for obs in observations:
            groups.setdefault(obs.symptom, []).append(obs)
        for symptom in sorted(groups):
            belief = None
            for obs in groups[symptom]:
                candidate = Belief(obs.stv, [("obs", obs.id)])
                belief, _ = merge_belief(belief, candidate)
            self.symptom_beliefs[symptom] = belief
            if len(groups[symptom]) > 1:
                premises = [f"report {o.id}: {self.patient.name} -> {symptom} {fmt(o.stv)}"
                            for o in groups[symptom]]
                self.trace.append(TraceEntry(
                    0, "revision", premises, f"{self.patient.name} -> {symptom}",
                    symptom, belief.stv, f"{len(groups[symptom])} reports of the same symptom merged"))
            self.store("symptom", symptom, belief)

    def abduce(self, symptom_name, symptom_belief, disease_name):
        """Abduction: Patient -> symptom and Disease -> symptom give Patient -> Disease."""
        disease = self.node(disease_name)
        symptom = self.node(symptom_name)
        observed = Atom("ImplicationLink", targets=(self.patient, symptom),
                        stv=symptom_belief.stv)
        kb_link = self.space.get("ImplicationLink", targets=[disease, symptom])
        absent = kb_link is None
        if absent:
            confidence = self.info.absent_confidence.get(disease_name)
            if confidence is None:
                return None
            kb_link = Atom("ImplicationLink", targets=(disease, symptom),
                           stv=STV(0.0, confidence))
            tag = ("kb-absent", disease_name, symptom_name)
        else:
            tag = ("kb", disease_name, symptom_name)
        atom = abduction_link(observed, kb_link)
        if atom is None:
            return None
        return Derivation("abduction", [observed, kb_link], atom,
                          symptom_belief.evidence | {tag}, absent)

    def deduce(self, disease_name, disease_belief, category_name):
        """Deduction: Patient -> Disease and Disease -> Category give Patient -> Category."""
        disease = self.node(disease_name)
        category = self.node(category_name)
        first = Atom("ImplicationLink", targets=(self.patient, disease),
                     stv=disease_belief.stv)
        second = self.space.get("InheritanceLink", targets=[disease, category])
        if second is None:
            return None
        atom = deduction_link(first, second)
        if atom is None:
            return None
        return Derivation("deduction", [first, second], atom,
                          disease_belief.evidence | {("kb", disease_name, category_name)})

    def accept(self, derivation):
        """Decide whether a derivation is kept. Returns (ok, reason)."""
        stv = derivation.atom.stv
        if is_no_info(stv):
            return False, "dropped: no information (confidence 0)"
        if stv.confidence < self.min_confidence:
            return False, f"dropped: confidence below {self.min_confidence}"
        return True, ""

    def suggest(self, disease_name, limit):
        """Induction: which unobserved symptoms of the disease are worth checking."""
        disease = self.node(disease_name)
        observed = set(self.symptom_beliefs)
        best = {}
        for o in sorted(observed):
            ba = self.space.get("ImplicationLink", targets=[disease, self.node(o)])
            if ba is None:
                continue
            for bc in self.space.links_from(disease, "ImplicationLink"):
                x = bc.targets[1]
                if x.name in observed:
                    continue
                result = induction_link(ba, bc)
                if result is None or is_no_info(result.stv):
                    continue
                if result.stv.confidence < self.min_confidence:
                    continue
                candidate = Suggestion(x.name, result.stv.strength,
                                       result.stv.confidence, o)
                current = best.get(x.name)
                if current is None or ((candidate.confidence, candidate.strength)
                                       > (current.confidence, current.strength)):
                    best[x.name] = candidate
        ranked = sorted(best.values(),
                        key=lambda s: (-s.strength, -s.confidence, s.symptom))
        return ranked[:limit]

    def prove_symptom(self, name, depth):
        goal = f"{self.patient.name} -> {name}"
        if depth > self.max_depth:
            return None, ProofNode(goal, "depth limit reached"), []
        belief = self.symptom_beliefs.get(name)
        if belief is None:
            return None, ProofNode(goal, "no evidence", note="symptom not reported"), []
        return belief, ProofNode(goal, "observation", belief.stv,
                                 note=f"evidence {sorted(belief.evidence)}"), []

    def prove_disease(self, name, depth):
        goal = f"{self.patient.name} -> {name}"
        if depth > self.max_depth:
            return None, ProofNode(goal, "depth limit reached"), []
        kb_symptoms = self.info.disease_symptoms[name]
        belief = None
        used = 0
        children = []
        missing = []
        for symptom in sorted(set(kb_symptoms) | set(self.symptom_beliefs)):
            s_belief, s_node, _ = self.prove_symptom(symptom, depth + 1)
            if s_belief is None:
                if symptom in kb_symptoms and depth + 1 <= self.max_depth:
                    link = kb_symptoms[symptom]
                    missing.append(Missing(symptom, link.stv.strength,
                                           link.stv.confidence, name))
                continue
            derivation = self.abduce(symptom, s_belief, name)
            if derivation is None:
                continue
            kb_atom = derivation.premises[1]
            kb_node = ProofNode(
                f"{name} -> {symptom}",
                "assumed absent" if derivation.absent else "knowledge base",
                kb_atom.stv,
                note="never seen with this disease: strength 0" if derivation.absent else "")
            ok, why = self.accept(derivation)
            children.append(ProofNode(goal, "abduction", derivation.atom.stv,
                                      [s_node, kb_node], note=why))
            if not ok:
                continue
            belief, _ = merge_belief(belief, Belief(derivation.atom.stv, derivation.evidence))
            used += 1
        if belief is None:
            note = "no usable derivation (no observed symptoms to reason from)"
            return None, ProofNode(goal, "abduction", None, children, note), missing
        rule = "revision of abductions" if used > 1 else "abduction"
        return belief, ProofNode(goal, rule, belief.stv, children), missing

    def prove_category(self, name, depth):
        goal = f"{self.patient.name} -> {name}"
        if depth > self.max_depth:
            return None, ProofNode(goal, "depth limit reached"), []
        belief = None
        children = []
        step_nodes = {}
        paths = {}                      # disease -> (evidence, missing list)
        all_missing = {}
        for disease in self.info.category_diseases[name]:
            d_belief, d_node, d_missing = self.prove_disease(disease, depth + 1)
            for m in d_missing:
                current = all_missing.get(m.symptom)
                if current is None or m.strength > current.strength:
                    all_missing[m.symptom] = m
            if d_belief is None:
                children.append(d_node)
                continue
            derivation = self.deduce(disease, d_belief, name)
            if derivation is None:
                continue
            kb_node = ProofNode(f"{disease} -> {name}", "knowledge base",
                                derivation.premises[1].stv)
            ok, why = self.accept(derivation)
            step = ProofNode(goal, "deduction", derivation.atom.stv,
                             [d_node, kb_node], note=why)
            children.append(step)
            if not ok:
                continue
            step_nodes[disease] = step
            paths[disease] = (derivation.evidence, d_missing)
            belief, _ = merge_belief(belief, Belief(derivation.atom.stv, derivation.evidence))
        if belief is None:
            return None, ProofNode(goal, "deduction", None, children,
                                   "no disease of this category could be proved"), []
        winner = None
        for disease in sorted(paths):
            if paths[disease][0] == belief.evidence:
                winner = disease
        for disease, step in step_nodes.items():
            if disease == winner:
                step.note = "best path (kept)"
            else:
                step.note = "not used: shares evidence with the best path, lower confidence"
        missing = paths[winner][1] if winner is not None else list(all_missing.values())
        return belief, ProofNode(goal, "best path of deductions", belief.stv, children,
                                 f"via {winner}" if winner else ""), missing

class ForwardResult:
    """Everything forward chaining produced."""

    def __init__(self, patient, observations, beliefs, suggestions, trace, rounds, space):
        self.patient = patient
        self.observations = observations      # list of (symptom, STV)
        self.beliefs = beliefs                # (kind, name) -> Belief
        self.suggestions = suggestions
        self.trace = trace
        self.rounds = rounds
        self.space = space
        self.diseases = self._ranked("disease")
        self.categories = self._ranked("category")

    def _ranked(self, kind):
        rows = [Ranked(name, b.stv.strength, b.stv.confidence)
                for (k, name), b in self.beliefs.items() if k == kind]
        rows.sort(key=lambda r: (-r.strength, -r.confidence, r.name))
        return rows

    def get(self, name):
        """Truth value of Patient -> name, or None if there is no belief."""
        for (_, n), belief in self.beliefs.items():
            if n == name:
                return belief.stv
        return None

    def top_disease(self):
        return self.diseases[0] if self.diseases else None

    def trace_text(self, target=None):
        entries = [e for e in self.trace if target is None or e.target == target]
        return "\n".join(str(e) for e in entries)

    def report(self, top=7):
        lines = ["Observations:"]
        if not self.observations:
            lines.append("  (none)")
        for symptom, stv in self.observations:
            lines.append(f"  {symptom}: strength {stv.strength:.4f}, confidence {stv.confidence:.4f}")
        lines.append("Diseases (ranked by strength, then confidence):")
        if not self.diseases:
            lines.append("  (no belief could be derived)")
        for i, r in enumerate(self.diseases[:top], 1):
            lines.append(f"  {i}. {r.name}: strength {r.strength:.4f}, confidence {r.confidence:.4f}")
        lines.append("Categories:")
        if not self.categories:
            lines.append("  (none)")
        for i, r in enumerate(self.categories, 1):
            lines.append(f"  {i}. {r.name}: strength {r.strength:.4f}, confidence {r.confidence:.4f}")
        if self.suggestions:
            top_name = self.top_disease().name
            lines.append(f"Symptoms worth checking next (induction from {top_name}):")
            for s in self.suggestions:
                lines.append(f"  {s.symptom}: strength {s.strength:.4f}, "
                             f"confidence {s.confidence:.4f} (given {s.via})")
        lines.append(f"Rounds that derived something: {self.rounds}")
        return "\n".join(lines)

class BackwardResult:
    """Everything backward chaining produced for one goal."""

    def __init__(self, patient, goal, kind, stv, proof, missing):
        self.patient = patient
        self.goal = goal
        self.kind = kind
        self.stv = stv
        self.proof = proof
        self.missing = missing

    def report(self):
        lines = [f"Goal: {self.patient} -> {self.goal}  ({self.kind})"]
        if self.stv is None:
            lines.append("Result: no belief could be proved")
        else:
            lines.append(f"Result: strength {self.stv.strength:.4f}, "
                         f"confidence {self.stv.confidence:.4f}")
        lines.append("Proof:")
        lines.extend("  " + line for line in self.proof.lines())
        if self.missing:
            lines.append("Unobserved symptoms that would help (by strength of the link to the disease):")
            for m in self.missing:
                lines.append(f"  {m.symptom}: {m.disease} -> {m.symptom} "
                             f"strength {m.strength:.4f}")
        return "\n".join(lines)

def compare_results(before, after):
    """Text table of how each disease and category changed between two forward runs."""
    names = [r.name for r in after.diseases] + [r.name for r in after.categories]
    for r in before.diseases + before.categories:
        if r.name not in names:
            names.append(r.name)
    lines = [f"{'statement':<22}{'before':<20}{'after':<20}{'change in strength'}"]
    for name in names:
        b, a = before.get(name), after.get(name)
        b_text = fmt(b) if b else "-"
        a_text = fmt(a) if a else "-"
        if b and a:
            change = f"{a.strength - b.strength:+.4f}"
        else:
            change = "n/a"
        lines.append(f"{name:<22}{b_text:<20}{a_text:<20}{change}")
    return "\n".join(lines)

class DiagnosisEngine:
    """Holds the knowledge base and the patient's observations."""

    def __init__(self, kb, patient="Patient1"):
        self.kb = kb
        self.patient = patient
        self.info = _KBInfo(kb)
        if kb.get("ConceptNode", name=patient) is not None:
            raise ValueError(f"'{patient}' already exists in the knowledge base")
        self._observations = []
        self._next_id = 1

    # ---- observations ----------------------------------------------------

    def add_observation(self, symptom, strength=1.0, confidence=DEFAULT_OBSERVATION_CONFIDENCE):
        """Record 'patient has symptom'. Returns the observation id."""
        if symptom not in self.info._symptom_set:
            raise ValueError(f"unknown symptom '{symptom}'."
                             + _did_you_mean(symptom, self.info.symptoms))
        stv = STV(strength, confidence)
        obs = Observation(self._next_id, symptom, stv)
        self._next_id += 1
        self._observations.append(obs)
        return obs.id

    def remove_observation(self, obs_id):
        for i, obs in enumerate(self._observations):
            if obs.id == obs_id:
                del self._observations[i]
                return
        raise ValueError(f"no observation with id {obs_id}")

    def clear_observations(self):
        self._observations = []

    def observations(self):
        return list(self._observations)

    # ---- forward chaining ------------------------------------------------

    def forward_chain(self, max_rounds=MAX_ROUNDS, min_confidence=MIN_CONFIDENCE,
                      suggestion_limit=SUGGESTION_LIMIT):
        """Derive diseases and categories from the observations."""
        ws = _Workspace(self)
        ws.min_confidence = min_confidence
        info = self.info
        beliefs = {("symptom", s): b for s, b in ws.symptom_beliefs.items()}
        applied = {}
        rounds = 0

        for rnd in range(1, max_rounds + 1):
            snapshot = dict(beliefs)
            derived = []

            # Rule 1: abduction, observed symptoms -> diseases.
            for symptom in sorted(ws.symptom_beliefs):
                s_belief = snapshot[("symptom", symptom)]
                for disease in info.diseases:
                    key = ("abduction", symptom, disease)
                    if applied.get(key) == s_belief.evidence:
                        continue
                    applied[key] = s_belief.evidence
                    derived.append(("disease", disease,
                                    ws.abduce(symptom, s_belief, disease)))

            # Rule 2: deduction, diseases -> categories.
            for disease in info.diseases:
                d_belief = snapshot.get(("disease", disease))
                if d_belief is None:
                    continue
                for category in info.disease_categories[disease]:
                    key = ("deduction", disease, category)
                    if applied.get(key) == d_belief.evidence:
                        continue
                    applied[key] = d_belief.evidence
                    derived.append(("category", category,
                                    ws.deduce(disease, d_belief, category)))

            if not derived:
                break
            rounds = rnd
            changed = False
            for kind, name, derivation in derived:
                if derivation is None:
                    continue
                premises = [link_text(p) for p in derivation.premises]
                conclusion = f"{ws.patient.name} -> {name}"
                ok, why = ws.accept(derivation)
                if not ok:
                    ws.trace.append(TraceEntry(rnd, derivation.rule, premises, conclusion,
                                               name, derivation.atom.stv, why))
                    continue
                candidate = Belief(derivation.atom.stv, derivation.evidence)
                merged, action = merge_belief(beliefs.get((kind, name)), candidate)
                if action in CHANGING_ACTIONS:
                    beliefs[(kind, name)] = merged
                    ws.store(kind, name, merged)
                    changed = True
                note = action if action == "new" else f"{action}: belief is now {fmt(merged.stv)}"
                ws.trace.append(TraceEntry(rnd, derivation.rule, premises, conclusion,
                                           name, derivation.atom.stv, note))
            if not changed:
                break

        result_obs = [(s, ws.symptom_beliefs[s].stv) for s in sorted(ws.symptom_beliefs)]
        diseases = [(n, b) for (k, n), b in beliefs.items() if k == "disease"]
        suggestions = []
        if diseases:
            top = sorted(diseases, key=lambda x: (-x[1].stv.strength, -x[1].stv.confidence, x[0]))[0][0]
            suggestions = ws.suggest(top, suggestion_limit)
        return ForwardResult(self.patient, result_obs, beliefs, suggestions,
                             ws.trace, rounds, ws.space)

    # ---- backward chaining -----------------------------------------------

    def backward_chain(self, goal, max_depth=MAX_DEPTH, min_confidence=MIN_CONFIDENCE,
                       max_missing=SUGGESTION_LIMIT):
        """Try to prove 'Patient -> goal' (a disease, category or symptom)."""
        kind = self.info.kind(goal)
        ws = _Workspace(self)
        ws.max_depth = max_depth
        ws.min_confidence = min_confidence
        if kind == "symptom":
            belief, node, missing = ws.prove_symptom(goal, 0)
        elif kind == "disease":
            belief, node, missing = ws.prove_disease(goal, 0)
        else:
            belief, node, missing = ws.prove_category(goal, 0)
        missing = sorted(missing, key=lambda m: (-m.strength, m.symptom))[:max_missing]
        stv = belief.stv if belief is not None else None
        return BackwardResult(self.patient, goal, kind, stv, node, missing)