import pandas as pd, sys, re
from collections import Counter
sys.path.insert(0, ".")
from src.data.taxonomy import classify
df = pd.read_csv("data/raw/train.csv")
df["family"] = df["prompt"].apply(classify)
bm = df[df["family"] == "bit_manip"].reset_index(drop=True)
print(f"total bit_manip rows: {len(bm)}\n")

print("=== 5 full sample prompts ===\n")
for i in range(5):
    r = bm.iloc[i]
    print(f"--- sample {i+1}  id={r['id']}  answer={r['answer']!r} ---")
    print(r["prompt"])
    print()

print("\n=== answer length distribution ===")
bm["al"] = bm["answer"].astype(str).map(len)
print(bm["al"].describe())
print("\n=== answer character counts (top 10) ===")
cc = Counter()
for a in bm["answer"].astype(str):
    cc.update(a)
for ch,n in cc.most_common(10):
    print(f"  {ch!r}: {n}")

print("\n=== prompt lengths ===")
bm["pl"] = bm["prompt"].astype(str).map(len)
print(bm["pl"].describe())

print("\n=== operator keyword frequencies (substring count over all 1602 prompts) ===")
for kw in ["XOR","AND","OR","NOT","shift","rotate","majority","minority","parity","nand","nor","xnor","invert","complement","flip"]:
    pat = re.compile(re.escape(kw), re.IGNORECASE)
    c = sum(1 for p in bm["prompt"] if pat.search(p))
    if c > 0:
        print(f"  {kw}: {c}")

bm["ne"] = bm["prompt"].str.count(r"\n")
print("\n=== newlines per prompt (proxy for example count) ===")
print(bm["ne"].describe())
