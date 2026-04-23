import pandas as pd, sys
sys.path.insert(0, ".")
from src.data.taxonomy import classify
df = pd.read_csv("data/raw/train.csv")
df["family"] = df["prompt"].apply(classify)
print("FAMILY COUNTS (full train):")
print(df["family"].value_counts())
print("\n=== 2 sample train 'equations' prompts ===")
for _, row in df[df["family"]=="equations"].head(2).iterrows():
    print(f"--- id={row['id']} gold={row['answer']!r} ---")
    print(row["prompt"][:600])
    print()
