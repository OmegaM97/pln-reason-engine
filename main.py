import sys
from chaining import DiagnosisEngine, compare_results
from build_kb import build_kb, read_slice

LINE = "=" * 78


def add_all(engine, symptoms):
    """Each item is a name, or (name, strength, confidence)."""
    for item in symptoms:
        if isinstance(item, str):
            engine.add_observation(item)
        else:
            engine.add_observation(*item)


def show_forward(result, trace_targets):
    print("\n--- FORWARD CHAINING ---")
    print(result.report())
    for target in trace_targets:
        print(f"\nTrace for {target}:")
        print(result.trace_text(target))


def show_backward(engine, goals):
    print("\n--- BACKWARD CHAINING ---")
    for goal in goals:
        print()
        print(engine.backward_chain(goal).report())


def run(kb, title, symptoms, goals, trace_targets=()):
    """Standard scenario: fresh engine, observe, forward chain, backward chain."""
    print(f"\n{LINE}\n{title}\n{LINE}")
    engine = DiagnosisEngine(kb)
    add_all(engine, symptoms)
    result = engine.forward_chain()
    show_forward(result, trace_targets)
    show_backward(engine, goals)
    return result


def scenario_1(kb):
    return run(
        kb, "SCENARIO 1: evidence supports a conclusion",
        ["cough", "blood_in_sputum", "weight_loss", "sweating"],
        goals=["Tuberculosis", "BacterialInfection"])


def scenario_2(kb):
    return run(
        kb, "SCENARIO 2: incomplete and uncertain evidence",
        [("high_fever", 1.0, 0.6)],
        goals=["Dengue"])


def scenario_3(kb):
    return run(
        kb, "SCENARIO 3: several explanations are possible",
        ["high_fever", "headache", "nausea", "vomiting", "fatigue"],
        goals=["Typhoid", "Malaria"])


def scenario_4(kb):
    """New and conflicting evidence: run, add reports, run again, compare."""
    title = "SCENARIO 4: new and conflicting evidence changes a belief"
    print(f"\n{LINE}\n{title}\n{LINE}")
    engine = DiagnosisEngine(kb)
    add_all(engine, ["skin_rash", "itching", "high_fever", "malaise"])
    before = engine.forward_chain()
    print("\nBEFORE the new evidence")
    show_forward(before, [])

    add_all(engine, [("skin_rash", 0.1, 0.9), "pain_behind_the_eyes"])
    after = engine.forward_chain()
    print("\nAFTER the new evidence")
    show_forward(after, ["skin_rash", "Dengue"])

    print("\nChange in beliefs:")
    print(compare_results(before, after))
    show_backward(engine, ["Dengue", "Chicken pox"])
    return after


def scenario_5(kb):
    return run(
        kb, "SCENARIO 5: multi-step inference, confidence along the path",
        ["cough", "phlegm", "chest_pain", "breathlessness"],
        goals=["BacterialInfection"],
        trace_targets=["Pneumonia", "BacterialInfection"])


def print_summary(results):
    print(f"\n{LINE}\nSUMMARY (top disease and top category per scenario)\n{LINE}")
    print(f"{'scenario':<14}{'top disease':<16}{'str':<8}{'conf':<8}"
          f"{'top category':<20}{'str':<8}{'conf'}")
    for name, r in results.items():
        d = r.diseases[0] if r.diseases else None
        c = r.categories[0] if r.categories else None
        print(f"{name:<14}"
              f"{(d.name if d else '-'):<16}"
              f"{(f'{d.strength:.3f}' if d else '-'):<8}"
              f"{(f'{d.confidence:.3f}' if d else '-'):<8}"
              f"{(c.name if c else '-'):<20}"
              f"{(f'{c.strength:.3f}' if c else '-'):<8}"
              f"{(f'{c.confidence:.3f}' if c else '-')}")


def run_all(kb):
    results = {
        "1 supported": scenario_1(kb),
        "2 incomplete": scenario_2(kb),
        "3 multiple": scenario_3(kb),
        "4 conflict": scenario_4(kb),
        "5 multi-step": scenario_5(kb),
    }
    print_summary(results)
    return results


SCENARIOS = {
    "1": ("Scenario 1: Evidence supports a conclusion", scenario_1),
    "2": ("Scenario 2: Incomplete and uncertain evidence", scenario_2),
    "3": ("Scenario 3: Several explanations are possible", scenario_3),
    "4": ("Scenario 4: New and conflicting evidence changes a belief", scenario_4),
    "5": ("Scenario 5: Multi-step inference, confidence along the path", scenario_5),
    "6": ("Run all scenarios (with summary)", run_all),
}


def menu(kb):
    while True:
        print(f"\n{LINE}")
        print("PLN REASON ENGINE - SCENARIO MENU")
        print(LINE)
        for key, (label, _) in SCENARIOS.items():
            print(f"  [{key}] {label}")
        print("  [0] Exit (or 'q')")
        print(LINE)

        try:
            choice = input("Select a scenario to run: ").strip().lower()
        except (KeyboardInterrupt, EOFError):
            print("\nExiting.")
            break

        if choice in ("0", "q", "exit", "quit"):
            print("Exiting.")
            break
        elif choice in SCENARIOS:
            _, func = SCENARIOS[choice]
            func(kb)
        else:
            print(f"Invalid selection '{choice}'. Please select an option from the menu.")


def main():
    symptoms, rows = read_slice("fever_slice.csv")
    kb, _ = build_kb(symptoms, rows)

    if len(sys.argv) > 1:
        choice = sys.argv[1].strip().lower()
        if choice in SCENARIOS:
            SCENARIOS[choice][1](kb)
            return
        elif choice in ("all", "all_scenarios"):
            run_all(kb)
            return
        else:
            print(f"Unknown argument '{choice}'. Starting interactive menu.")

    menu(kb)


if __name__ == "__main__":
    main()