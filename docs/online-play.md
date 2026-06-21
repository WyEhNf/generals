# Online Play

The online adapter connects a local policy to bot.generals.io through Socket.IO.

## Install dependencies first

Online checkpoint play needs the full runtime dependencies, including `numpy`,
`python-socketio`, and `torch`:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

Run the online commands with the same Python environment. A
`ModuleNotFoundError` for `numpy`, `socketio`, or `torch` means this install
step was skipped or a different Python interpreter is being used.

## Required settings

- `GENERALS_AGENT`: local agent spec, usually `checkpoint:checkpoints/step1_step2_step3/model.pt`.
- `GENERALS_USER_ID`: private bot id from bot.generals.io. This is not the display name.
- `GENERALS_USERNAME`: display name to set for the bot.
- `GENERALS_ROOM_ID`: private room id for `GENERALS_MODE=private`.
- `GENERALS_DEVICE`: `cpu`, `cuda`, or `auto`.

## Private-room run

```bash
cp .env.example .env
PYTHONPATH=src .venv/bin/python scripts/run_remote_agent.py
```

The script writes an event log under `artifacts/remote_matches/` by default.

## Command-line run

```bash
PYTHONPATH=src .venv/bin/python scripts/run_remote_agent.py \
  --agent checkpoint:checkpoints/step1_step2_step3/model.pt \
  --user-id YOUR_PRIVATE_USER_ID \
  --username "[Bot] your_bot_name" \
  --room-id your_private_room \
  --mode private \
  --device cpu
```

Keep `.env` private. Do not commit real bot ids.
