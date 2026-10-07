# Mini PLN Disease Reason Engine

A small Probabilistic Logic Networks (PLN) engine in Python for diagnosing fever causing infectious diseases and their categories (bacterial, viral, parasitic) from reported patient symptoms. Every relationship and derived belief carries a Simple Truth Value (STV) with **strength** (probability) and **confidence** (amount of evidence).

---

## Repository Files

This project uses a flat layout (all core modules reside in the root):

- **`main.py`**: Interactive entry point running diagnosis scenarios and chain traces.
- **`chaining.py`**: Forward and backward reasoning engines with evidence tracking and loop detection.
- **`pln_rules.py`**: Core PLN probabilistic inference rules (Deduction, Abduction, Induction, Revision, Inversion).
- **`build_kb.py`**: Builds the AtomSpace knowledge base from the prepared CSV slice and exports to JSON.
- **`atomspace.py`**: In-memory hypergraph store for Concepts, Predicates, and Links with STVs.
- **`prepare_data.py`**: Script to clean Kaggle raw data into `fever_slice.csv`.
- **`data_preparation.ipynb`**: Interactive notebook explaining the raw data exploration, disease selection, and cleaning steps.
- **`data/`**: Raw Kaggle dataset files (`Training.csv`, `Testing.csv`).
- **`fever_slice.csv`**: Cleaned data slice (7 fever-related infectious diseases, 40 symptoms).
- **`kb.json`**: Serialized AtomSpace knowledge base.
- **`DOCUMENTATION.md`**: Detailed technical documentation covering formulas, AtomSpace architecture, knowledge base parameters, chaining algorithms, and scenario details.

> For in-depth implementation details (AtomSpace indexes, exact PLN math formulas, chaining rounds, evidence merging, and assumptions), see [DOCUMENTATION.md](DOCUMENTATION.md).

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
