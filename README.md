# Mini PLN Disease Reason Engine

A small Probabilistic Logic Networks (PLN) engine in Python for diagnosing fever causing infectious diseases and their categories (bacterial, viral, parasitic) from reported patient symptoms. Every relationship and derived belief carries a Simple Truth Value (STV) with **strength** (probability) and **confidence** (amount of evidence).

---

## Repository Structure & Key Documents

The project uses a flat layout (all core modules reside in the root). To explore the project, start with the primary documentation and data exploration notebook:

### Primary Documentation & Analysis (Start Here)

- **[`DOCUMENTATION.md`](DOCUMENTATION.md)**: **In-Depth Technical Reference** Complete breakdown of the reasoning architecture, exact PLN mathematical formulas (Deduction, Abduction, Induction, Revision, Inversion), AtomSpace indexing, chaining cycles, and scenario results.
- **[`data_preparation.ipynb`](data_preparation.ipynb)**: **Interactive Data Pipeline & Exploration** — Step-by-step walkthrough detailing how the raw Kaggle data was cleaned, why the 7 fever-causing diseases were chosen, symptom filtering, and dataset limitations (such as duplicate row analysis in Section 9).

---

### Core Reasoning Engine

- **`main.py`**: Interactive terminal entry point providing a menu to run clinical scenarios (1–5) or all scenarios with a comparison table.
- **`chaining.py`**: Forward and backward chaining engines featuring evidence tracking and circular-reasoning prevention.
- **`pln_rules.py`**: Formal implementations of PLN probabilistic rules operating on Simple Truth Values (strength and confidence).
- **`atomspace.py`**: In-memory hypergraph store for Concepts, Predicates, and Links with fast indexes.
- **`build_kb.py`**: Builds the AtomSpace knowledge base from `fever_slice.csv` and serializes it to `kb.json`.

---

### Data & Knowledge Base Artifacts

- **`prepare_data.py`**: Standalone script that cleans and slices the raw Kaggle dataset.
- **`fever_slice.csv`**: Cleaned data slice containing 7 fever-related infectious diseases and 40 active symptoms (847 total rows).
- **`data/`**: Raw Kaggle dataset files (`Training.csv` and `Testing.csv`).
- **`kb.json`**: Pre-built serialized AtomSpace knowledge base.

---

## How to Run End-to-End

Run the interactive scenario menu using `uv`:

```bash
uv run main.py
```

You can select individual clinical scenarios (1–5) or run all scenarios at once (6) with a comparative summary table.

You can also run specific scenarios directly via arguments:

```bash
uv run python main.py 1       # Run Scenario 1
uv run python main.py all     # Run all scenarios with summary
```

To rebuild the knowledge base from `fever_slice.csv`:

```bash
uv run python build_kb.py
```

Technical Report: https://docs.google.com/document/d/1AlJStZHcZsMcxxG7W11Vg9Qxs2ZsdKprIiBOTIHBvKk/edit?usp=sharing
Presentation Slide: https://docs.google.com/presentation/d/1WcB0qsa70YYhmWaaxP4LKkBxCQSoBydL/edit?usp=sharing&ouid=117691053317499707814&rtpof=true&sd=true