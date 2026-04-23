import os, time, sys, pandas as pd
sys.path.insert(0, ".")
from dotenv import load_dotenv
from openai import OpenAI
from src.eval.metric import extract_answer, is_correct
load_dotenv()
client = OpenAI(base_url=os.environ["TEACHER_BASE_URL"], api_key=os.environ["TEACHER_API_KEY"], timeout=120.0, max_retries=0)
model = os.environ["TEACHER_MODEL"]
print(f"Model: {model}", flush=True)
df = pd.read_csv("data/raw/train.csv").head(3)
SYS = ("You are a careful reasoning assistant solving few-shot rule-induction puzzles. "
       "Think step by step: identify the hidden rule from the examples, verify it on every "
       "example, then apply it to the query. End your response with \\boxed{ANSWER} containing "
       "ONLY the bare answer — no units, no LaTeX commands, no quotes.")
results = []
for i, (_, r) in enumerate(df.iterrows()):
    print(f"\n[{i+1}/3] id={r['id']} calling API...", flush=True)
    t0 = time.time()
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role":"system","content":SYS},{"role":"user","content":r["prompt"]}],
            temperature=0.0, max_tokens=8000,
            extra_body={"reasoning": {"effort": "low"}},
        )
        dt = time.time() - t0
        content = resp.choices[0].message.content or ""
        ans = extract_answer(content)
        ok = is_correct(ans, str(r["answer"]))
        results.append(ok)
        usage = resp.usage
        rt = getattr(getattr(usage, 'completion_tokens_details', None), 'reasoning_tokens', 'n/a')
        print(f"  elapsed={dt:.1f}s completion={usage.completion_tokens} reasoning={rt} prompt={usage.prompt_tokens}", flush=True)
        print(f"  teacher={ans!r}  gold={r['answer']!r}  correct={ok}", flush=True)
        print(f"  tail: ...{content[-200:]}", flush=True)
    except Exception as e:
        dt = time.time() - t0
        print(f"  ERROR after {dt:.1f}s: {type(e).__name__}: {e}", flush=True)

print(f"\nFINAL: {sum(results)}/{len(results)} correct")
