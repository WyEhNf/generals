# Checkpoints

Only the report-facing checkpoints are included in this release. Each directory
contains a single `model.pt` file and is named after the method in the report.

| Release name | Meaning |
| --- | --- |
| `bc` | Behavior cloning initialization |
| `terminal_ppo` | Terminal win/loss PPO from BC |
| `high_city` | Naive high city-reward ablation |
| `step1` | Stage 1: keep city behavior while learning combat |
| `step1_step2` | Stage 1+2: increase city-use signal while preserving strength |
| `step1_step2_step3` | Final balanced staged city-aware agent |

Use `checkpoint:checkpoints/step1_step2_step3/model.pt` for the final agent.

The checkpoint files are included for immediate local evaluation. If you want a
lighter Git history, upload the `model.pt` files as GitHub Release assets or use
Git LFS before the first public commit.
