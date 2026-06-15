# Model Card

This release contains six checkpoints aligned with the report's method names.

| Model | Path | Intended use |
|---|---|---|
| BC | `checkpoints/bc/model.pt` | Supervised behavior-cloning baseline |
| Terminal PPO | `checkpoints/terminal_ppo/model.pt` | PPO baseline using only terminal win/loss reward |
| High city | `checkpoints/high_city/model.pt` | Naive city-reward ablation |
| Step 1 | `checkpoints/step1/model.pt` | Curriculum stage 1 |
| Step 1+2 | `checkpoints/step1_step2/model.pt` | Curriculum stages 1 and 2 |
| Step 1+2+3 | `checkpoints/step1_step2_step3/model.pt` | Final balanced agent |

The final model is `Step 1+2+3`. It is the default model in `.env.example`.

These checkpoints are for local simulation, visualization, evaluation, and online private-room play. They are not enough to reproduce training from scratch without replay data and training logs.
