import json, collections
from pathlib import Path
p = Path("data/distill/teacher_traces.jsonl")
rows = [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]
print(f"TOTAL ROWS: {len(rows)}")
kept = [r for r in rows if r.get("kept")]
print(f"KEPT: {len(kept)} ({len(kept)/len(rows)*100:.1f}%)")
by_fam = collections.defaultdict(lambda: [0,0])
for r in rows:
    by_fam[r["family"]][0] += int(bool(r.get("kept")))
    by_fam[r["family"]][1] += 1
print("\nPER-FAMILY kept/total:")
for fam, (k, t) in sorted(by_fam.items()):
    print(f"  {fam:<12s} {k:>3d}/{t:<3d}  ({k/t*100:.0f}%)")
lat = [r["latency_s"] for r in rows if r.get("latency_s")]
if lat:
    ls = sorted(lat)
    print(f"\nLATENCY (s): min={min(lat):.1f} med={ls[len(ls)//2]:.1f} p95={ls[int(len(ls)*0.95)]:.1f} max={max(lat):.1f}")
errs = [r.get("error") for r in rows if r.get("error")]
print(f"\nERRORS: {len(errs)}")
for e in errs[:5]:
    print(f"  {str(e)[:160]}")
print("\nSAMPLE FAILURES (non-error):")
fails = [r for r in rows if not r.get("kept") and not r.get("error")]
for r in fails[:8]:
    print(f"  [{r['family']:<12s}] teacher={r['teacher_answer']!r:<40s} gold={r['gold']!r}")
