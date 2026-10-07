# Mini PLN Engine for Fever Diagnosis

A small Probabilistic Logic Networks (PLN) engine in Python. It diagnoses the likely cause of a fever from reported symptoms, using truth values (strength, confidence) on every relationship.

## 1. Problem and why probabilistic reasoning

**Problem.** Given the symptoms of one patient, rank the possible infectious diseases and their categories (viral, bacterial, parasitic), say how sure the system is, and say which symptom to check next.

**Why probabilistic.** Symptoms are shared between diseases (fever, headache, fatigue appear in most of them), so no single symptom decides. Evidence is partial and can conflict. Each relationship therefore carries two numbers: **strength** (how often it holds) and **confidence** (how much evidence supports that strength).

## 2. AtomSpace (`atomspace.py`)

The knowledge is stored as atoms

- **STV(strength, confidence)**: both validated to be in [0, 1]. Default `(1.0, 0.0)` means "no information".
- **Atom**: a node (type + name) or a link (type + ordered tuple of target atoms). Identity is `(type, name or targets)`; the STV is **not** part of identity, so one statement has one truth value.
- Node types: `ConceptNode`, `PredicateNode`. Link types: `InheritanceLink`, `SubsetLink`, `SimilarityLink`, `EvaluationLink`, `ImplicationLink`. Only Concept, Predicate, Implication, Inheritance and Evaluation are used; Subset and Similarity are defined but unused.
- **AtomSpace**: a dict `key -> atom` plus indexes by type, by first target, by last target and by containing link.
  - `add(atom)` stores an atom once; if it exists, only the STV is updated. Missing link targets are added too.
  - `get(type, name=... | targets=[...])` finds one exact atom, or returns `None`.
  - `of_type(type)` lists atoms of a type. `links_from(atom, type)` lists links whose **first** target is the atom. `links_to` uses the last target. `match` does simple pattern queries. `remove` deletes an atom and the links containing it.
  - `save(path)` / `load(path)` write and read JSON (every atom with its type, name or targets, and STV).

## 3. Knowledge base construction (`knowledge_base.py`)

`read_slice(path)` checks the file (first two columns are `prognosis` and `category`, symptom cells are 0 or 1, no empty values) and returns the symptom names and rows. `build_kb(symptoms, rows)` returns the AtomSpace and a stats dict.

| Atom | Meaning | Strength | Confidence |
|---|---|---|---|
| Disease `ConceptNode` | Disease is the diagnosis (base rate) | 1/7 (assumed equal) | 0.1 |
| Category `ConceptNode` | Category (base rate) | (number of member diseases) / 7 | 0.1 |
| Symptom `ConceptNode` | Symptom is present (base rate) | rows with symptom / all rows | n/(n+K), n = 847 |
| `ImplicationLink` Disease → Symptom | Fraction of the disease's rows that show the symptom | count / 121 | n/(n+K), n = 121 |
| `InheritanceLink` Disease → Category | The disease is a kind of the category | 1.0 (Pneumonia: 0.7) | 0.9 |

- **Confidence from evidence count:** `c = n / (n + K)` with **K = 10**. With n = 121 this is 0.9237.
- Symptoms never seen with a disease get no link (count 0 is skipped).
- Because every disease has the same number of rows, a symptom's base rate equals the average of its disease link strengths, so base rates and link strengths agree. This matters for deduction and abduction.
- The saved JSON is reloaded and compared atom by atom before the build is reported as successful.

## 4. PLN rules (`pln_rules.py`)

Notation for a link A→B: `sAB`, `cAB`. `sB` is the base rate (strength) of node B. All rules return both strength and confidence.

| Rule | Use in this system | Premises → conclusion |
|---|---|---|
| Abduction | Symptom → disease (diagnosis) | Patient→Symptom, Disease→Symptom ⟹ Patient→Disease |
| Deduction | Disease → category | Patient→Disease, Disease→Category ⟹ Patient→Category |
| Revision | Merge results about the same statement | two truth values for one statement ⟹ one |
| Induction | "Check next" suggestions | Disease→Symptom1, Disease→Symptom2 ⟹ Symptom1→Symptom2 |

**Deduction** (A→B, B→C ⟹ A→C):
- Strength: `sAC = sAB·sBC + (1 − sAB)·(sC − sB·sBC) / (1 − sB)`
- Confidence: `cAC = cAB · cBC · sAB · sBC`

**Abduction** (A→B, C→B ⟹ A→C; B is shared target, e.g. A = patient, B = symptom, C = disease):
- Strength: `sAC = (sAB·sCB·sC / sB) + sC·(1 − sAB)·(1 − sCB) / (1 − sB)`
- Confidence: `x = sAB · cAB · cCB`, then `cAC = x / (x + 1)`

**Induction** (B→A, B→C ⟹ A→C; B is shared source, e.g. B = disease, A = symptom 1, C = symptom 2):
- Strength: `sAC = (sBA·sBC·sB / sA) + (1 − sBA·sB / sA)·(sC − sB·sBC) / (1 − sB)`
- Confidence: `x = sBC · cBC · cBA`, then `cAC = x / (x + 1)`

**Revision** (two truth values `(s1, c1)` and `(s2, c2)` for the same statement; evidence weights `w1 = c1 / (1 − c1)`, `w2 = c2 / (1 − c2)`, `w = w1 + w2`):
- Strength: `s = (w1·s1 + w2·s2) / w` (if `w = 0`, `s = (s1 + s2) / 2`)
- Confidence: `c = max(w / (w + 1), c1, c2)`

**Inversion** (A→B ⟹ B→A; not used in chaining):
- Strength: `sBA = sAB · sA / sB`
- Confidence: `cBA = cB · cAB · 0.6`

**Deduction over mixed link types.** The reference PLN library (`lib_pln.metta`) only chains two links of the same type. Here the first link is an ImplicationLink (Patient→Disease) and the second is an InheritanceLink (Disease→Category). The deduction here accepts this mix and returns an ImplicationLink. The category links stay InheritanceLinks. The formula is unchanged; only the type check differs.

**No-information results.** A result with confidence 0 carries no evidence. The engine drops it, and also drops any result with confidence below 0.01.

## 5. Chaining (`chaining.py`)

Both modes run on a **copy** of the knowledge base (`copy_space`), so scenarios do not affect each other. Both use the same helper functions (`abduce`, `deduce`) and the same evidence rules, so the same goal gives the same numbers.

**Observations.** `add_observation(symptom, strength=1.0, confidence=0.9)` records "Patient1 has symptom". The engine stores it as an `EvaluationLink(has_symptom, Patient1, symptom)` and as an `ImplicationLink Patient1 → symptom`, because the rules chain only Inheritance and Implication links. Patient1 is a node with base rate (0.01, 0.1). Absence of a symptom is not modelled.

**Evidence IDs.** Each observation and each knowledge base link has an evidence ID. A derived belief carries the union of its premises' IDs. When two results about the same statement meet:
1. already counted (same or fewer IDs): keep the existing one;
2. disjoint evidence: merge with revision;
3. overlapping evidence: keep the one with higher confidence.

This prevents the same evidence from being counted twice. Several reports of the same symptom are independent evidence and are merged by revision first.

**Missing links.** If a disease has no link to an observed symptom, it is treated as strength 0 (the symptom was never seen with it), with the confidence of that disease's strongest link. This lowers the disease.

**Forward chaining** (`forward_chain()`), data-driven:
1. Round 1, abduction: every observed symptom × every disease gives Patient→Disease. Results for one disease from different symptoms are merged by revision.
2. Round 2, deduction: every disease belief × its category gives Patient→Category. When several diseases lead to one category, the best path (higher confidence) is kept, because those paths share the observation evidence.
3. Stops when nothing new appears, or after 3 rounds.
4. Induction on the top disease suggests unobserved symptoms to check.

Output: diseases and categories ranked by strength, then confidence; the suggestions; and a trace of every step (rule, premises, result).

**Backward chaining** (`backward_chain(goal)`), goal-driven. The goal is a disease, a category or a symptom.
- Category: needs deduction, so its subgoals are the diseases of that category.
- Disease: needs abduction, so its subgoals are the observed symptoms plus the knowledge base links.
- Symptom: proved by an observation, otherwise "no evidence".
- Depth limit 3. Output: strength, confidence, the proof tree, and the unobserved symptoms that would help (missing evidence).

## 6. Scenarios (`main.py`)

Each scenario uses a fresh engine over the same knowledge base. Observations have strength 1.0 and confidence 0.9 unless stated.

| # | Observations | Top disease (s, c) | Top category (s, c) |
|---|---|---|---|
| 1 Evidence supports a conclusion | cough, blood_in_sputum, weight_loss, sweating | Tuberculosis (0.667, 0.769) | Bacterial (0.778, 0.461) |
| 2 Incomplete evidence | high_fever, confidence 0.6 | Typhoid (0.149, 0.357) | Bacterial (0.433, 0.048) |
| 3 Several explanations | high_fever, headache, nausea, vomiting, fatigue | Dengue (0.222, 0.806) | Viral (0.481, 0.161) |
| 4 New or conflicting evidence | see below | Chicken pox (0.330, 0.792) | Viral (0.553, 0.235) |
| 5 Multi-step | cough, phlegm, chest_pain, breathlessness | Tuberculosis (0.376, 0.769) | Bacterial (0.584, 0.260) |

**1.** Tuberculosis leads (Pneumonia 0.167). Four symptoms agree, and revision raises the confidence to 0.769. The strength is 0.667 because it averages the four abduction results (1.0, 0.33, 0.33, 1.0). Forward and backward chaining give the same values.

**2.** One uncertain symptom cannot separate the diseases: all seven sit between 0.14 and 0.15 with confidence 0.357. The "check next" list and the missing-evidence list show what to ask.

**3.** Dengue (0.222) and Typhoid (0.220) are almost tied, and Malaria follows (0.184). Viral (0.481) and Bacterial (0.480) are almost tied as categories.

**4.** Before: skin_rash, itching, high_fever, malaise gives Chicken pox (0.462, 0.769) and Dengue (0.209, 0.769). Then a second report of skin_rash with strength 0.1 and a new symptom pain_behind_the_eyes are added. The two rash reports merge to (0.55, 0.947); Chicken pox falls to 0.330 and Dengue rises to 0.328. The lead of Chicken pox shrinks from 0.25 to 0.002.

**5.** Confidence along the path to BacterialInfection: observation 0.90, each abduction 0.454, the revision of four symptoms 0.769 (Tuberculosis), the category by deduction 0.260. Confidence falls at each inference step and rises when independent evidence is merged.

## 7. Assumptions

1. Each row is treated as one independent patient (n = 121 per disease). See limitation 3.
2. K = 10 in `c = n / (n + K)`.
3. Disease base rate 1/7 (confidence 0.1): the file has equal rows per disease, so real frequencies are unknown. Category base rate is the sum of its diseases' base rates.
4. Category links: confidence 0.9; strength 1.0, except Pneumonia → BacterialInfection at 0.7 (hand-set, because it can also be viral). The base rate of Bacterial (3/7) is not adjusted for this.
5. Observation confidence defaults to 0.9. Patient1 base rate (0.01, 0.1).
6. A symptom never seen with a disease is treated as strength 0 for that disease.
7. Absence of a symptom is not modelled.
8. Thresholds: results with confidence below 0.01 are dropped; at most 3 rounds; depth limit 3.
