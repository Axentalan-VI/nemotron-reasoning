# Progress

## Current status

Finished, **4157 of 4185 teams**, one submission scoring **0.384 accuracy**.
The competition closed on 2026-06-15. 57 tests pass.

The engineering worked: chain-of-thought traces distilled from a teacher model,
Nemotron-3-Nano-30B QLoRA-finetuned in 4-bit (rank <= 32) on a single H100, and
the adapter served through vLLM inside the Kaggle inference limits. The model
was never tuned past that first adapter.

## Last session (2026-10-06)

- Corrected the README, which claimed the competition was still running.
- Recorded the submission in `EXPERIMENTS.md`.
- For the record: the competition slug in the README was wrong for a while
  (`nvidia-open-models`, which 404s). The correct one is
  `nvidia-nemotron-model-reasoning-challenge`, and the wrong slug had been
  hiding this submission entirely.

## Open issues

- **One submission, near the bottom of the board.** The serving pipeline is the
  deliverable; the accuracy is not.
- No local evaluation score was recorded, so there is no way to know whether
  the distillation helped at all.

## Next steps (prioritized)

1. If revisited: measure accuracy locally on a held-out slice of the distilled
   traces before spending a submission.

## Decisions & rationale

- QLoRA at rank <= 32 in 4-bit: the competition caps the adapter rank, and 30B
  in 4-bit is what fits a single H100 (2026-05).
- A vLLM-compatible adapter rather than a merged model, to stay inside the
  host's inference time limit (2026-05).
