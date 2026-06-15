from __future__ import annotations

import json
from pathlib import Path

from generals_bot.agents.base import EnemyKingDistribution, WinRateEstimate
from generals_bot.sim.types import Action, ExecutedAction, Observation, PlayerScore
from generals_bot.viz.recorder import Snapshot


def render_html(snapshots: list[Snapshot]) -> str:
    data = [_snapshot_to_dict(snapshot) for snapshot in snapshots]
    encoded = json.dumps(data, separators=(",", ":"))
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<link rel="icon" href="data:,">
<title>Generals Local Replay</title>
<style>
:root {{
  color-scheme: light;
  font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
}}
body {{
  margin: 0;
  background: #f6f7f9;
  color: #1f2933;
}}
main {{
  display: grid;
  grid-template-columns: minmax(320px, 1fr) 340px;
  gap: 20px;
  min-height: 100vh;
  padding: 20px;
  box-sizing: border-box;
}}
.boards {{
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(320px, 1fr));
  gap: 16px;
  align-content: start;
}}
.board-panel {{
  background: white;
  border: 1px solid #d7dce2;
  border-radius: 8px;
  padding: 12px;
  min-width: 0;
}}
.board-title {{
  display: flex;
  align-items: baseline;
  justify-content: space-between;
  gap: 8px;
  margin-bottom: 10px;
  font-size: 14px;
  font-weight: 800;
}}
.board-note {{
  color: #52606d;
  font-size: 12px;
  font-weight: 600;
  text-align: right;
  white-space: nowrap;
}}
.board {{
  display: grid;
  align-content: start;
  justify-content: center;
  gap: 2px;
  overflow: auto;
}}
.cell {{
  width: 24px;
  height: 24px;
  display: grid;
  place-items: center;
  border: 1px solid #111827;
  box-sizing: border-box;
  font-size: 11px;
  font-weight: 700;
  position: relative;
}}
.p0 {{ background: #ef4444; color: white; }}
.p1 {{ background: #2563eb; color: white; }}
.neutral {{ background: #d7dce2; color: #1f2933; }}
.mountain {{ background: #6b7280; color: white; }}
.fog {{ background: #111827; color: #111827; }}
.fog-obstacle {{
  background: repeating-linear-gradient(
    135deg,
    #111827,
    #111827 5px,
    #374151 5px,
    #374151 10px
  );
  color: #e5e7eb;
}}
.city::after {{
  content: "C";
  position: absolute;
  right: 2px;
  top: 1px;
  font-size: 9px;
}}
.general::before,
.known-general::before {{
  content: "G";
  position: absolute;
  left: 2px;
  top: 1px;
  font-size: 9px;
}}
.known-enemy-general::before {{
  content: "E";
}}
.probability-title {{
  margin: 12px 0 6px;
  color: #52606d;
  font-size: 12px;
  font-weight: 800;
}}
.probability-map {{
  display: grid;
  align-content: start;
  justify-content: center;
  gap: 2px;
  overflow: auto;
}}
.probability-cell {{
  width: 24px;
  height: 24px;
  display: grid;
  place-items: center;
  border: 1px solid #cbd5e1;
  box-sizing: border-box;
  color: #111827;
  font-size: 9px;
  font-weight: 800;
  position: relative;
}}
.probability-cell.max-probability {{
  border: 2px solid #111827;
}}
.probability-empty {{
  color: #697586;
  font-size: 12px;
  font-weight: 700;
  padding: 8px 0 2px;
  text-align: center;
}}
aside {{
  background: white;
  border: 1px solid #d7dce2;
  border-radius: 8px;
  padding: 16px;
  align-self: start;
}}
.controls {{
  display: grid;
  grid-template-columns: 36px 36px 1fr 36px;
  gap: 8px;
  align-items: center;
  margin-bottom: 16px;
}}
button {{
  height: 32px;
  border: 1px solid #9aa5b1;
  background: #fff;
  border-radius: 6px;
  cursor: pointer;
}}
select {{
  height: 32px;
  border: 1px solid #9aa5b1;
  background: #fff;
  border-radius: 6px;
}}
input[type="range"] {{ width: 100%; }}
.scoreboard {{
  display: grid;
  gap: 10px;
  margin-bottom: 16px;
}}
.turn-line {{
  font-size: 20px;
  font-weight: 800;
}}
.score-row {{
  display: grid;
  grid-template-columns: 72px 1fr 1fr;
  gap: 8px;
  align-items: center;
  font-size: 13px;
}}
.player-label {{
  color: #fff;
  border-radius: 4px;
  padding: 5px 6px;
  font-weight: 800;
  text-align: center;
}}
.label-p0 {{ background: #ef4444; }}
.label-p1 {{ background: #2563eb; }}
.metric {{
  border: 1px solid #d7dce2;
  border-radius: 6px;
  padding: 5px 6px;
  background: #f8fafc;
}}
.playback {{
  display: grid;
  grid-template-columns: 1fr 96px;
  gap: 8px;
  margin-bottom: 16px;
}}
pre {{
  white-space: pre-wrap;
  font-size: 12px;
  line-height: 1.35;
}}
@media (max-width: 760px) {{
  main {{ grid-template-columns: 1fr; }}
  aside {{ order: -1; }}
}}
</style>
</head>
<body>
<main>
  <section class="boards" aria-label="boards">
    <div class="board-panel">
      <div class="board-title">
        <span>Global</span>
        <span class="board-note">truth</span>
      </div>
      <div id="global-board" class="board"></div>
    </div>
    <div class="board-panel">
      <div class="board-title">
        <span>P0 POV</span>
        <span id="p0-win-rate" class="board-note">N/A</span>
      </div>
      <div id="p0-board" class="board"></div>
      <div class="probability-title">Enemy king probability</div>
      <div id="p0-king-probability" class="probability-map"></div>
    </div>
    <div class="board-panel">
      <div class="board-title">
        <span>P1 POV</span>
        <span id="p1-win-rate" class="board-note">N/A</span>
      </div>
      <div id="p1-board" class="board"></div>
      <div class="probability-title">Enemy king probability</div>
      <div id="p1-king-probability" class="probability-map"></div>
    </div>
  </section>
  <aside>
    <div class="scoreboard" aria-label="scoreboard">
      <div class="turn-line">Turn <span id="current-turn">0</span></div>
      <div class="score-row">
        <span class="player-label label-p0">P0</span>
        <span class="metric">Army <strong id="p0-army">0</strong></span>
        <span class="metric">Land <strong id="p0-land">0</strong></span>
      </div>
      <div class="score-row">
        <span class="player-label label-p1">P1</span>
        <span class="metric">Army <strong id="p1-army">0</strong></span>
        <span class="metric">Land <strong id="p1-land">0</strong></span>
      </div>
    </div>
    <div class="controls">
      <button id="prev" type="button">-</button>
      <button id="play" type="button" title="Play or pause">&gt;</button>
      <input id="turn" type="range" min="0" max="0" value="0">
      <button id="next" type="button">+</button>
    </div>
    <div class="playback">
      <span>Playback</span>
      <select id="playback-rate" aria-label="playback rate">
        <option value="1000">1x</option>
        <option value="500" selected>2x</option>
        <option value="200">5x</option>
        <option value="100">10x</option>
      </select>
    </div>
    <pre id="meta"></pre>
  </aside>
</main>
<script>
const snapshots = {encoded};
const boards = {{
  global: document.getElementById("global-board"),
  0: document.getElementById("p0-board"),
  1: document.getElementById("p1-board"),
}};
const probabilityMaps = {{
  0: document.getElementById("p0-king-probability"),
  1: document.getElementById("p1-king-probability"),
}};
const slider = document.getElementById("turn");
const meta = document.getElementById("meta");
const prev = document.getElementById("prev");
const next = document.getElementById("next");
const play = document.getElementById("play");
const playbackRate = document.getElementById("playback-rate");
const currentTurn = document.getElementById("current-turn");
const p0Army = document.getElementById("p0-army");
const p0Land = document.getElementById("p0-land");
const p1Army = document.getElementById("p1-army");
const p1Land = document.getElementById("p1-land");
const p0WinRate = document.getElementById("p0-win-rate");
const p1WinRate = document.getElementById("p1-win-rate");
let playbackTimer = null;
slider.max = Math.max(0, snapshots.length - 1);

function globalCellClass(owner) {{
  if (owner === 0) return "p0";
  if (owner === 1) return "p1";
  if (owner === -2) return "mountain";
  return "neutral";
}}

function observationCellClass(owner, playerId) {{
  if (owner === 0) return playerId === 0 ? "p0" : "p1";
  if (owner === 1) return playerId === 0 ? "p1" : "p0";
  if (owner === -2) return "mountain";
  if (owner === -3) return "fog";
  if (owner === -4) return "fog-obstacle";
  return "neutral";
}}

function setBoardColumns(container, width) {{
  container.style.gridTemplateColumns = `repeat(${{width}}, 24px)`;
}}

function appendCell(container, className, army) {{
  const cell = document.createElement("div");
  cell.className = `cell ${{className}}`;
  cell.textContent = army > 0 ? String(army) : "";
  container.appendChild(cell);
  return cell;
}}

function formatWinRate(estimate) {{
  if (estimate === null || estimate === undefined) return "N/A";
  const pct = (estimate.win_probability * 100).toFixed(1);
  const raw = estimate.raw_value.toFixed(3);
  return `Win ${{pct}}% | V=${{raw}}`;
}}

function formatProbability(probability) {{
  if (probability <= 0.00005) return "";
  if (probability < 0.01) return "<1";
  return (probability * 100).toFixed(1);
}}

function probabilityCellColor(probability, maxProbability) {{
  if (maxProbability <= 0 || probability <= 0) return "#f8fafc";
  const normalized = Math.min(1, probability / maxProbability);
  const alpha = 0.10 + 0.82 * normalized;
  return `rgba(220, 38, 38, ${{alpha.toFixed(3)}})`;
}}

function probabilitySummary(distribution) {{
  if (distribution === null || distribution === undefined) return "N/A";
  const probabilities = distribution.probabilities;
  let maxProbability = -1;
  let maxRow = 0;
  let maxCol = 0;
  for (let row = 0; row < probabilities.length; row += 1) {{
    for (let col = 0; col < probabilities[row].length; col += 1) {{
      const probability = probabilities[row][col];
      if (probability > maxProbability) {{
        maxProbability = probability;
        maxRow = row;
        maxCol = col;
      }}
    }}
  }}
  return `max=${{(maxProbability * 100).toFixed(2)}}% @ (${{maxRow}},${{maxCol}})`;
}}

function renderGlobalBoard(s) {{
  const container = boards.global;
  setBoardColumns(container, s.width);
  container.textContent = "";
  for (let i = 0; i < s.width * s.height; i += 1) {{
    const row = Math.floor(i / s.width);
    const col = i % s.width;
    const owner = s.terrain[row][col];
    const cell = appendCell(container, globalCellClass(owner), s.armies[row][col]);
    if (s.cities[row][col]) cell.classList.add("city");
    if (s.generals.includes(i)) cell.classList.add("general");
  }}
}}

function renderObservationBoard(s, playerId) {{
  const obs = s.observations[String(playerId)];
  const container = boards[playerId];
  setBoardColumns(container, obs.width);
  container.textContent = "";
  for (let i = 0; i < obs.width * obs.height; i += 1) {{
    const row = Math.floor(i / obs.width);
    const col = i % obs.width;
    const owner = obs.owner[row][col];
    const cell = appendCell(
      container,
      observationCellClass(owner, playerId),
      obs.armies[row][col],
    );
    if (obs.cities[row][col]) cell.classList.add("city");
    if (obs.known_generals[row][col]) cell.classList.add("known-general");
    if (obs.known_enemy_generals[row][col]) cell.classList.add("known-enemy-general");
  }}
}}

function renderProbabilityMap(s, playerId) {{
  const distribution = s.enemy_king_distributions[String(playerId)];
  const obs = s.observations[String(playerId)];
  const container = probabilityMaps[playerId];
  setBoardColumns(container, obs.width);
  container.textContent = "";
  if (distribution === null || distribution === undefined) {{
    const empty = document.createElement("div");
    empty.className = "probability-empty";
    empty.textContent = "N/A";
    empty.style.gridColumn = `span ${{obs.width}}`;
    container.appendChild(empty);
    return;
  }}
  const probabilities = distribution.probabilities;
  let maxProbability = 0;
  let maxRow = 0;
  let maxCol = 0;
  for (let row = 0; row < obs.height; row += 1) {{
    for (let col = 0; col < obs.width; col += 1) {{
      const probability = probabilities[row][col];
      if (probability > maxProbability) {{
        maxProbability = probability;
        maxRow = row;
        maxCol = col;
      }}
    }}
  }}
  for (let row = 0; row < obs.height; row += 1) {{
    for (let col = 0; col < obs.width; col += 1) {{
      const probability = probabilities[row][col];
      const cell = document.createElement("div");
      cell.className = "probability-cell";
      if (row === maxRow && col === maxCol) cell.classList.add("max-probability");
      cell.style.backgroundColor = probabilityCellColor(probability, maxProbability);
      cell.textContent = formatProbability(probability);
      cell.title = `row=${{row}} col=${{col}} probability=${{(probability * 100).toFixed(4)}}%`;
      container.appendChild(cell);
    }}
  }}
}}

function render(index) {{
  const s = snapshots[index];
  const p0 = s.scores.find(score => score.player_id === 0);
  const p1 = s.scores.find(score => score.player_id === 1);
  renderGlobalBoard(s);
  renderObservationBoard(s, 0);
  renderObservationBoard(s, 1);
  renderProbabilityMap(s, 0);
  renderProbabilityMap(s, 1);
  currentTurn.textContent = String(s.turn);
  p0Army.textContent = String(p0 ? p0.army : 0);
  p0Land.textContent = String(p0 ? p0.land : 0);
  p1Army.textContent = String(p1 ? p1.army : 0);
  p1Land.textContent = String(p1 ? p1.land : 0);
  p0WinRate.textContent = formatWinRate(s.win_rates["0"]);
  p1WinRate.textContent = formatWinRate(s.win_rates["1"]);
  meta.textContent = [
    `snapshot: ${{index + 1}} / ${{snapshots.length}}`,
    `winner: ${{s.winner === null ? "-" : s.winner}}`,
    "",
    "scores:",
    ...s.scores.map(score => `p${{score.player_id}} army=${{score.army}} land=${{score.land}} dead=${{score.dead}}`),
    "",
    "observations:",
    `p0 own=${{s.observations["0"].own_army}}/${{s.observations["0"].own_land}} enemy=${{s.observations["0"].enemy_army}}/${{s.observations["0"].enemy_land}}`,
    `p1 own=${{s.observations["1"].own_army}}/${{s.observations["1"].own_land}} enemy=${{s.observations["1"].enemy_army}}/${{s.observations["1"].enemy_land}}`,
    "",
    "value head:",
    `p0 ${{formatWinRate(s.win_rates["0"])}}`,
    `p1 ${{formatWinRate(s.win_rates["1"])}}`,
    "",
    "enemy king probability:",
    `p0 ${{probabilitySummary(s.enemy_king_distributions["0"])}}`,
    `p1 ${{probabilitySummary(s.enemy_king_distributions["1"])}}`,
    "",
    "submitted:",
    JSON.stringify(s.submitted_actions),
    "",
    "executed:",
    JSON.stringify(s.executed_actions),
    "",
    "queue:",
    JSON.stringify(s.queued_actions),
  ].join("\\n");
}}

function setIndex(index) {{
  slider.value = String(index);
  render(index);
}}

function stopPlayback() {{
  if (playbackTimer !== null) {{
    window.clearInterval(playbackTimer);
    playbackTimer = null;
  }}
  play.textContent = ">";
}}

function startPlayback() {{
  if (Number(slider.value) >= snapshots.length - 1) {{
    setIndex(0);
  }}
  play.textContent = "||";
  playbackTimer = window.setInterval(() => {{
    const nextIndex = Number(slider.value) + 1;
    if (nextIndex >= snapshots.length) {{
      stopPlayback();
      return;
    }}
    setIndex(nextIndex);
  }}, Number(playbackRate.value));
}}

slider.addEventListener("input", () => {{
  stopPlayback();
  render(Number(slider.value));
}});
prev.addEventListener("click", () => {{
  stopPlayback();
  setIndex(Math.max(0, Number(slider.value) - 1));
}});
next.addEventListener("click", () => {{
  stopPlayback();
  setIndex(Math.min(snapshots.length - 1, Number(slider.value) + 1));
}});
play.addEventListener("click", () => {{
  if (playbackTimer === null) {{
    startPlayback();
  }} else {{
    stopPlayback();
  }}
}});
playbackRate.addEventListener("change", () => {{
  if (playbackTimer !== null) {{
    stopPlayback();
    startPlayback();
  }}
}});
render(0);
</script>
</body>
</html>
"""


def write_html(snapshots: list[Snapshot], path: str | Path) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(render_html(snapshots), encoding="utf-8")


def _snapshot_to_dict(snapshot: Snapshot) -> dict:
    return {
        "turn": int(snapshot.turn),
        "width": int(snapshot.width),
        "height": int(snapshot.height),
        "terrain": snapshot.terrain.tolist(),
        "armies": snapshot.armies.tolist(),
        "cities": snapshot.cities.tolist(),
        "generals": [int(general) for general in snapshot.generals],
        "alive": [bool(alive) for alive in snapshot.alive],
        "queued_actions": _actions_by_player(snapshot.queued_actions),
        "submitted_actions": _actions_by_player(snapshot.submitted_actions),
        "executed_actions": [_executed_to_dict(action) for action in snapshot.executed_actions],
        "scores": [_score_to_dict(score) for score in snapshot.scores],
        "winner": None if snapshot.winner is None else int(snapshot.winner),
        "observations": {
            str(player): _observation_to_dict(observation)
            for player, observation in snapshot.observations.items()
        },
        "win_rates": {
            str(player): _win_rate_to_dict(snapshot.win_rates.get(player))
            for player in (0, 1)
        },
        "enemy_king_distributions": {
            str(player): _enemy_king_distribution_to_dict(
                snapshot.enemy_king_distributions.get(player)
            )
            for player in (0, 1)
        },
    }


def _actions_by_player(actions: dict[int, list[Action]]) -> dict[str, list[dict]]:
    return {
        str(player): [_action_to_dict(action) for action in player_actions]
        for player, player_actions in actions.items()
    }


def _action_to_dict(action: Action) -> dict:
    return {
        "player_id": int(action.player_id),
        "start": int(action.start),
        "end": int(action.end),
        "split": bool(action.split),
    }


def _executed_to_dict(executed: ExecutedAction) -> dict:
    return {
        "player_id": int(executed.player_id),
        "action": _action_to_dict(executed.action),
        "turn": int(executed.turn),
        "valid": bool(executed.valid),
        "reason": executed.reason,
        "captured_general": (
            None if executed.captured_general is None else int(executed.captured_general)
        ),
    }


def _score_to_dict(score: PlayerScore) -> dict:
    return {
        "player_id": int(score.player_id),
        "army": int(score.army),
        "land": int(score.land),
        "dead": bool(score.dead),
        "has_kill": bool(score.has_kill),
    }


def _observation_to_dict(observation: Observation) -> dict:
    return {
        "player_id": int(observation.player_id),
        "turn": int(observation.turn),
        "width": int(observation.width),
        "height": int(observation.height),
        "visible": observation.visible.tolist(),
        "explored": observation.explored.tolist(),
        "armies": observation.armies.tolist(),
        "owner": observation.owner.tolist(),
        "own_tiles": observation.own_tiles.tolist(),
        "enemy_tiles": observation.enemy_tiles.tolist(),
        "neutral_tiles": observation.neutral_tiles.tolist(),
        "mountains": observation.mountains.tolist(),
        "cities": observation.cities.tolist(),
        "generals": observation.generals.tolist(),
        "known_generals": observation.known_generals.tolist(),
        "known_enemy_generals": observation.known_enemy_generals.tolist(),
        "fog": observation.fog.tolist(),
        "fog_obstacles": observation.fog_obstacles.tolist(),
        "own_army": int(observation.own_army),
        "own_land": int(observation.own_land),
        "enemy_army": int(observation.enemy_army),
        "enemy_land": int(observation.enemy_land),
        "last_moves": [
            {
                "player_id": int(move.player_id),
                "start": int(move.start),
                "end": int(move.end),
                "split": bool(move.split),
                "turn": int(move.turn),
                "visible": bool(move.visible),
            }
            for move in observation.last_moves
        ],
        "priority": int(observation.priority),
    }


def _win_rate_to_dict(estimate: WinRateEstimate | None) -> dict | None:
    if estimate is None:
        return None
    return {
        "win_probability": float(estimate.win_probability),
        "raw_value": float(estimate.raw_value),
    }


def _enemy_king_distribution_to_dict(
    estimate: EnemyKingDistribution | None,
) -> dict | None:
    """Convert an optional enemy king distribution to JSON-friendly data."""
    if estimate is None:
        return None
    return {"probabilities": estimate.probabilities.tolist()}
