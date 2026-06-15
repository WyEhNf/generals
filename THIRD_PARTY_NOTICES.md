# Third-Party Notices

This repository is a student research / engineering project for training and
evaluating Generals.io agents. It is not an official Generals.io project.

## Flobot benchmark port

`src/generals_bot/benchmark_bots/flobot/` is a Python port of
`@corsaircoalition/flobot` v6.0.0.

Original sources:

- https://www.npmjs.com/package/@corsaircoalition/flobot
- https://github.com/CorsairCoalition/Flobot

Original license: Apache-2.0.

The port keeps the original module structure and strategy behavior where it is
meaningful in this local simulator, while exposing the project-local
`Policy.reset()` / `Policy.act()` interface.

## Replay dataset

Behavior-cloning experiments used the public Generals.io replay dataset:

- https://huggingface.co/datasets/strakammm/generals_io_replays

The raw replay dataset is not included in this release package.

## External services

The online adapter connects to `bot.generals.io` for private-room bot testing.
Private bot IDs, usernames, room IDs, and `.env` files are intentionally excluded
from this repository.

## Python dependencies

The project depends on open-source Python packages such as NumPy, PyTorch,
python-socketio, tqdm, and TensorBoard. See `requirements.txt` and
`requirements-train.txt` for the runtime and training dependencies.
