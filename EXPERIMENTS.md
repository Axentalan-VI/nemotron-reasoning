# Experiments

The one submission to the
[NVIDIA Nemotron Model Reasoning Challenge](https://www.kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge).
Metric: accuracy, higher is better.

| ID | Date | Change | Public LB | Notes |
|----|------|--------|-----------|-------|
| E001 | 2026-05-21 | QLoRA adapter, distilled traces, vLLM | 0.384 | only submission |

No local evaluation was logged, so this number has nothing to be compared
against: not the base model, not a prompt-only baseline. Either comparison
would have said whether the fine-tune was worth its GPU hours.
