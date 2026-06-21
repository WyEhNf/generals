# Reproducibility Commands

Install runtime dependencies first:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

Use the same `.venv/bin/python` interpreter for every command below.

## Render a single match

```bash
PYTHONPATH=src .venv/bin/python scripts/render_match.py \
  --agent0 checkpoint:checkpoints/step1_step2_step3/model.pt \
  --agent1 checkpoint:checkpoints/terminal_ppo/model.pt \
  --turns 800 \
  --map generated \
  --min-size 18 \
  --max-size 18 \
  --seed 0 \
  --device cpu \
  --output artifacts/demo_final_vs_terminal.html \
  --report artifacts/demo_final_vs_terminal.json
```

## Batch head-to-head evaluation

```bash
PYTHONPATH=src .venv/bin/python scripts/evaluate_match.py \
  --agent final=checkpoint:checkpoints/step1_step2_step3/model.pt \
  --agent terminal=checkpoint:checkpoints/terminal_ppo/model.pt \
  --pair final:terminal \
  --turns 800 \
  --seed-start 0 \
  --seed-count 200 \
  --map generated \
  --min-size 18 \
  --max-size 18 \
  --batched \
  --device cpu \
  --output-dir artifacts/eval_final_vs_terminal
```

For quick checks, reduce `--seed-count` to 20.

## City behavior evaluation

```bash
PYTHONPATH=src .venv/bin/python scripts/evaluate_city_behavior.py \
  --agent final=checkpoint:checkpoints/step1_step2_step3/model.pt \
  --mode selfplay \
  --target-turn 1200 \
  --seed-start 0 \
  --seed-count 200 \
  --map generated \
  --min-size 18 \
  --max-size 18 \
  --device cpu \
  --output-dir artifacts/city_behavior_final

PYTHONPATH=src .venv/bin/python scripts/summarize_city_capture_outcome.py \
  --input artifacts/city_behavior_final/records.jsonl \
  --output artifacts/city_behavior_final/city_capture_outcome.json
```
