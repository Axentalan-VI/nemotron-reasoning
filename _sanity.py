import os, asyncio, time, pandas as pd, sys
sys.path.insert(0, ".")
from dotenv import load_dotenv
from openai import AsyncOpenAI
from src.data.teacher_distill import _call_teacher, SYSTEM_PROMPT
from src.eval.metric import extract_answer, is_correct
load_dotenv()

async def main():
    client = AsyncOpenAI(base_url=os.environ["TEACHER_BASE_URL"], api_key=os.environ["TEACHER_API_KEY"], timeout=240.0, max_retries=0)
    model = os.environ["TEACHER_MODEL"]
    print(f"Model: {model}")
    df = pd.read_csv("data/raw/train.csv").head(3)
    for _, r in df.iterrows():
        t0 = time.time()
        try:
            text = await _call_teacher(client, model, r["prompt"], 8000)
            dt = time.time() - t0
            ans = extract_answer(text)
            ok = is_correct(ans, str(r["answer"]))
            tail = text[-200:].replace("\n", " | ")
            print(f"  id={r['id']} elapsed={dt:.1f}s  teacher={ans!r} gold={r['answer']!r} correct={ok}")
            print(f"    tail: ...{tail}")
        except Exception as e:
            print(f"  id={r['id']} ERROR after {time.time()-t0:.1f}s: {type(e).__name__}: {e}")
asyncio.run(main())
