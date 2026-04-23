import pandas as pd, sys, collections, re
sys.path.insert(0, ".")
from src.data.taxonomy import classify
df = pd.read_csv("data/raw/train.csv")
df["family"] = df["prompt"].apply(classify)
eq = df[df["family"] == "equations"].reset_index(drop=True)
print(f"total equations rows: {len(eq)}")
print("\n=== 6 full sample prompts with answers ===\n")
for i in range(6):
    r = eq.iloc[i]
    print(f"--- sample {i+1}  id={r['id']}  answer={r['answer']!r} ---")
    print(r["prompt"])
    print()

eq["ans_len"] = eq["answer"].astype(str).map(len)
print("\n=== answer length distribution ===")
print(eq["ans_len"].describe())
print("\n=== answer charset (top 40 chars) ===")
from collections import Counter
c = Counter()
for a in eq["answer"].astype(str):
    c.update(a)
for ch, n in c.most_common(40):
    print(f"  {ch!r}: {n}")
print("\n=== prompt length (chars) ===")
eq["p_len"] = eq["prompt"].astype(str).map(len)
print(eq["p_len"].describe())
print("\n=== examples-per-prompt (lines containing =) ===")
eq["n_eq_lines"] = eq["prompt"].str.count(r"=")
print(eq["n_eq_lines"].describe())
