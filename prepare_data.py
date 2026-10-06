import sys
import pandas as pd

TRAIN_INPUT = sys.argv[1] if len(sys.argv) > 1 else "data/Training.csv"
TEST_INPUT = sys.argv[2] if len(sys.argv) > 2 else "data/Testing.csv"

OUTPUT = "data/fever_slice.csv"

CATEGORY = {
    "Malaria": "ParasiticInfection",
    "Dengue": "ViralInfection",
    "Chicken pox": "ViralInfection",
    "Common Cold": "ViralInfection",
    "Typhoid": "BacterialInfection",
    "Tuberculosis": "BacterialInfection",
    "Pneumonia": "BacterialInfection",
}

MIN_COUNT = 1

train = pd.read_csv(TRAIN_INPUT)
test = pd.read_csv(TEST_INPUT)

df = pd.concat([train, test], ignore_index=True)

df.columns = [c.strip().replace(" ", "_") for c in df.columns]

df = df.loc[:, ~df.columns.str.startswith("Unnamed")]
df = df.drop(columns=["fluid_overload.1"], errors="ignore")

df["prognosis"] = df["prognosis"].str.strip()

df = df[df["prognosis"].isin(CATEGORY)].copy()
symptoms = [c for c in df.columns if c != "prognosis"]
keep = [s for s in symptoms if df[s].sum() >= MIN_COUNT]
df["category"] = df["prognosis"].map(CATEGORY)
out = df[["prognosis", "category"] + keep]

out.to_csv(OUTPUT, index=False)

print(f"Saved {OUTPUT}: {len(out)} rows, {len(keep)} symptoms kept")
print("\nDisease counts:")
print(out["prognosis"].value_counts().to_string())