import csv
import sys
from collections import OrderedDict

from atomspace import AtomSpace, Concept, Link, STV

K = 10                         
DISEASE_BASE_CONFIDENCE = 0.1
CATEGORY_LINK_CONFIDENCE = 0.9
CATEGORY_STRENGTH_OVERRIDES = {
    "Pneumonia": 0.7,
}

DEFAULT_INPUT = "fever_slice.csv"
DEFAULT_OUTPUT = "kb.json"


def confidence_from_count(n, k=K):
    """Confidence grows with the number of observations: n / (n + k)."""
    return n / (n + k)


def read_slice(path):
    """Return (symptom_names, rows) where each row is (disease, category, flags).

    flags is a list of 0/1 values in the order of symptom_names.
    """
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        try:
            header = next(reader)
        except StopIteration:
            raise ValueError(f"{path} is empty")

        header = [h.strip() for h in header]
        if len(header) < 3 or header[0] != "prognosis" or header[1] != "category":
            raise ValueError(
                "first two columns must be 'prognosis' and 'category', "
                f"got {header[:2]}"
            )
        symptoms = header[2:]
        if len(set(symptoms)) != len(symptoms):
            raise ValueError("duplicate symptom column names")

        rows = []
        for line_no, raw in enumerate(reader, start=2):
            if not raw or all(cell.strip() == "" for cell in raw):
                continue
            if len(raw) != len(header):
                raise ValueError(
                    f"line {line_no}: expected {len(header)} columns, got {len(raw)}"
                )
            disease = raw[0].strip()
            category = raw[1].strip()
            if not disease or not category:
                raise ValueError(f"line {line_no}: empty disease or category")
            flags = []
            for name, cell in zip(symptoms, raw[2:]):
                cell = cell.strip()
                if cell not in ("0", "1", "0.0", "1.0"):
                    raise ValueError(
                        f"line {line_no}: symptom '{name}' must be 0 or 1, got '{cell}'"
                    )
                flags.append(int(float(cell)))
            rows.append((disease, category, flags))

    if not rows:
        raise ValueError(f"{path} has no data rows")
    return symptoms, rows

def build_kb(symptoms, rows):
    """Build and return (AtomSpace, stats dict)."""
    total = len(rows)

    # Per-disease row counts and symptom counts.
    disease_rows = OrderedDict()                 # disease -> n rows
    disease_symptom = OrderedDict()              # disease -> [count per symptom]
    disease_category = OrderedDict()             # disease -> category
    symptom_total = [0] * len(symptoms)

    for disease, category, flags in rows:
        if disease not in disease_rows:
            disease_rows[disease] = 0
            disease_symptom[disease] = [0] * len(symptoms)
            disease_category[disease] = category
        elif disease_category[disease] != category:
            raise ValueError(
                f"disease '{disease}' has two categories: "
                f"'{disease_category[disease]}' and '{category}'"
            )
        disease_rows[disease] += 1
        for i, flag in enumerate(flags):
            disease_symptom[disease][i] += flag
            symptom_total[i] += flag

    diseases = list(disease_rows)
    categories = list(OrderedDict.fromkeys(disease_category.values()))
    disease_base = 1.0 / len(diseases)

    space = AtomSpace()

    # Disease nodes: equal assumed base rate.
    for d in diseases:
        space.add(Concept(d, STV(disease_base, DISEASE_BASE_CONFIDENCE)))

    # Category nodes: sum of the base rates of their diseases.
    for c in categories:
        members = [d for d in diseases if disease_category[d] == c]
        strength = min(1.0, disease_base * len(members))
        space.add(Concept(c, STV(strength, DISEASE_BASE_CONFIDENCE)))

    # Symptom nodes: frequency over all rows. Symptoms never seen are skipped.
    kept_symptoms = []
    for i, s in enumerate(symptoms):
        if symptom_total[i] == 0:
            continue
        kept_symptoms.append(s)
        space.add(
            Concept(s, STV(symptom_total[i] / total, confidence_from_count(total)))
        )

    # Disease -> Symptom ImplicationLinks.
    symptom_links = 0
    for d in diseases:
        n = disease_rows[d]
        conf = confidence_from_count(n)
        d_atom = space.get("ConceptNode", name=d)
        for i, s in enumerate(symptoms):
            count = disease_symptom[d][i]
            if count == 0:
                continue
            s_atom = space.get("ConceptNode", name=s)
            space.add(Link("ImplicationLink", d_atom, s_atom,
                           stv=STV(count / n, conf)))
            symptom_links += 1

    # Disease -> Category InheritanceLinks.
    for d in diseases:
        c_atom = space.get("ConceptNode", name=disease_category[d])
        d_atom = space.get("ConceptNode", name=d)
        strength = CATEGORY_STRENGTH_OVERRIDES.get(d, 1.0)
        space.add(Link("InheritanceLink", d_atom, c_atom,
                       stv=STV(strength, CATEGORY_LINK_CONFIDENCE)))

    stats = {
        "rows": total,
        "diseases": len(diseases),
        "categories": len(categories),
        "symptoms_in_file": len(symptoms),
        "symptoms_kept": len(kept_symptoms),
        "implication_links": symptom_links,
        "inheritance_links": len(diseases),
        "atoms": len(space),
        "rows_per_disease": dict(disease_rows),
    }
    return space, stats

def print_summary(space, stats):
    print(f"Rows read:              {stats['rows']}")
    print(f"Diseases:               {stats['diseases']}  {stats['rows_per_disease']}")
    print(f"Categories:             {stats['categories']}")
    print(f"Symptoms in file/kept:  {stats['symptoms_in_file']}/{stats['symptoms_kept']}")
    print(f"ImplicationLinks:       {stats['implication_links']}")
    print(f"InheritanceLinks:       {stats['inheritance_links']}")
    print(f"Total atoms:            {stats['atoms']}")

    print("\nDisease base rates and categories:")
    for link in sorted(space.of_type("InheritanceLink"), key=lambda l: l.targets[0].name):
        d, c = link.targets
        print(f"  {d.name:<14} base {d.stv}   -> {c.name:<18} link {link.stv}")

    print("\nCategory base rates:")
    for c in sorted(
        {l.targets[1] for l in space.of_type("InheritanceLink")},
        key=lambda a: a.name,
    ):
        print(f"  {c.name:<18} {c.stv}")

    print("\nStrongest Disease -> Symptom links per disease (top 3):")
    for d in sorted(
        {l.targets[0] for l in space.of_type("InheritanceLink")},
        key=lambda a: a.name,
    ):
        links = sorted(
            space.links_from(d, "ImplicationLink"),
            key=lambda l: (-l.stv.strength, l.targets[1].name),
        )[:3]
        text = ", ".join(f"{l.targets[1].name} {l.stv.strength:.2f}" for l in links)
        print(f"  {d.name:<14} {text}")


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    in_path = argv[0] if len(argv) > 0 else DEFAULT_INPUT
    out_path = argv[1] if len(argv) > 1 else DEFAULT_OUTPUT

    symptoms, rows = read_slice(in_path)
    space, stats = build_kb(symptoms, rows)
    space.save(out_path)

    reloaded = AtomSpace.load(out_path)
    if len(reloaded) != len(space):
        raise RuntimeError("saved knowledge base does not reload to the same size")
    for atom in space.all():
        other = reloaded.atoms.get(atom.key())
        if other is None or other.stv != atom.stv:
            raise RuntimeError(f"reload mismatch for {atom}")

    print_summary(space, stats)
    print(f"\nSaved {len(space)} atoms to {out_path} (reload check passed).")


if __name__ == "__main__":
    main()