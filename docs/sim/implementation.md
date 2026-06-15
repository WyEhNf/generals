# 模拟器实现计划

## 模块拆分

第一批只实现模拟器，不碰训练模型：

```text
src/generals_bot/
└── sim/
    ├── types.py      # Action, Observation, PlayerScore, GameConfig, terrain enum
    ├── state.py      # GameState, PlayerMemory, 初始化接口
    ├── queue.py      # ActionQueue、连续动作和非法动作跳过
    ├── mapgen.py     # 本地自博弈地图生成器
    ├── rules.py      # move / attack / capture / growth / win condition
    ├── observe.py    # 从全局状态生成玩家 POV observation
    └── env.py        # reset / step / submit / tick / observe
```

第二批做 replay：

```text
src/generals_bot/
├── replay/
│   ├── schema.py
│   ├── loader.py
│   └── replayer.py
└── training/
    └── samples.py
```

第三批做 agent 和工具：

```text
src/generals_bot/
├── agents/
│   ├── base.py
│   ├── expander.py
│   └── march.py
└── viz/
    ├── recorder.py
    └── html.py

scripts/
├── run_local_match.py
└── render_local_replay.py
```

线上 adapter 最后做，避免 websocket 的不稳定性污染模拟器设计。

## 全局状态

模拟器内部维护完整真值：

```python
@dataclass
class GameState:
    turn: int
    width: int
    height: int
    armies: np.ndarray       # int[H, W]
    terrain: np.ndarray      # owner id 或 terrain enum
    cities: np.ndarray       # bool[H, W]
    generals: list[int]
    alive: list[bool]
    queues: dict[int, ActionQueue]
    memory: dict[int, PlayerMemory]
```

`PlayerMemory` 维护每个玩家的 explored、已见过的 city/general/obstacle，用于生成带记忆的 observation。其中 `generals` observation 字段只表示当前可见将军，`known_generals` / `known_enemy_generals` 表示曾经见过且仍存活的将军位置。

## 实现顺序

建议按这个顺序落地：

```text
1. sim/types.py
   定义 Action, PassAction, Observation, PlayerScore, GameConfig, Terrain enum

2. sim/state.py
   定义 GameState, PlayerMemory, 从固定地图初始化

3. sim/queue.py
   实现 submit、pending queue、非法动作跳过

4. sim/rules.py
   实现移动、攻击、占领、自然增长、胜负检查

5. sim/mapgen.py
   实现 1v1 vanilla 地图生成、连通性校验、简单 spawn/city fairness 过滤

6. sim/observe.py
   从全局状态生成玩家 POV observation

7. sim/env.py
   封装 reset / step / submit / tick / observe

8. viz/
   实现本地模拟回放 recorder 和 HTML 可视化

9. tests/
   先写规则、队列、视野测试

10. replay/
   接入 replay schema 和 replay-to-samples
```

## 本地回放可视化

第一阶段需要配套一个本地模拟回放可视化工具，方便调试规则、动作队列、fog observation 和 bot 决策。

可视化分两层：

```text
viz/recorder.py
  从 GeneralsEnv 每个 tick 记录 Snapshot

viz/html.py
  把 snapshots 渲染成单文件 HTML
```

Snapshot 建议包含：

```python
@dataclass(frozen=True)
class Snapshot:
    turn: int
    terrain: np.ndarray
    armies: np.ndarray
    cities: np.ndarray
    generals: list[int]
    alive: list[bool]
    queued_actions: dict[int, list[Action]]
    submitted_actions: dict[int, list[Action]]
    executed_actions: list[ExecutedAction]
    scores: list[PlayerScore]
```

HTML 可视化第一版只做本地调试，不追求 UI 完整：

```text
- 单文件 HTML，直接浏览器打开
- 进度条 / 上一回合 / 下一回合
- 自动播放 / 暂停 / 播放速度
- 状态栏显示当前 turn、双方 army、land、dead
- 同时显示 3 个同步棋盘：Global 真值、P0 POV、P1 POV
- 颜色区分玩家、neutral、mountain、city、general
- 格子上显示 army
- 侧边栏显示 turn、army、land、队列长度、最后执行动作
- P0/P1 POV 直接来自 simulator 的 Observation，不在前端重新推导视野
- POV 画面必须保留 fog / fog obstacle，不泄漏不可见格兵力
- 如果 agent 提供 value head / win-rate estimate，POV 标题显示 Win% 和原始 V；
  rule-based agent 等不提供估计时显示 N/A
- 如果 agent 提供 enemy king probability map，POV 棋盘下方显示同尺寸 heatmap；
  每个格子显示概率数字，并用颜色强度突出当前帧更可能的位置
- 如果 agent 不提供概率图，或 checkpoint 缺少已训练的 `enemy_king_head` 权重，
  heatmap 区域显示 N/A，不显示随机初始化的概率
- checkpoint agent 可以通过 `enemy_king_predictor=<path>` 使用额外的 standalone
  predictor 生成概率图；这只影响 snapshot / HTML，不影响该 agent 的行动和 value
  估计
```

脚本接口：

```bash
python3 scripts/render_local_replay.py \
  --bot0 expander \
  --bot1 expander \
  --turns 200 \
  --seed 0 \
  --output artifacts/local_replay.html
```

当前脚本默认使用 `sim/mapgen.py` 的随机地图生成器，也保留 `--map demo` 用于回归固定小地图。

可视化的定位：

```text
- 用来检查模拟器规则是否符合直觉
- 用来定位 action timing / queue skip 的问题
- 用来查看 replay 重放偏差发生在哪个 turn
- 不作为训练数据格式
```

## 测试计划

最小测试集：

```text
test_move_capture:
  攻击中立格、敌方格、己方格的兵力变化

test_split_move:
  50% move 的移动兵力正确

test_illegal_move_skip:
  非法动作不会改变状态，队列继续处理后续动作

test_action_timing:
  obs_t 产生的动作会在本 tick 结算，结果进入 obs_{t+1}

test_growth:
  general/city 每 2 half-turn 增长，所有 owned land 每 50 half-turn 增长

test_fog_observation:
  不可见 army 不泄露，fog obstacle 正确显示

test_general_capture:
  capture general 后立即胜负正确

test_replay_smoke:
  任选 10 个 replay 能完整重放并生成样本

test_visualization_smoke:
  跑一局短本地对战并生成 HTML，确认包含关键 snapshot 数据

test_map_generation:
  随机生成多张地图，确认尺寸、general 距离、city army、连通性、可复现性
```

## Golden Test

需要从真实 replay 或线上日志建立 golden test，用来校准：

```text
- replay move.turn 的语义
- action submit turn 和可见生效 turn 的关系
- 线上 game_speed / RTT / tick 相位造成的额外可见延迟
- turn=0 / city 当 tick 占领等自然增长边界
- general capture 后领地和兵力转移，若后续需要 replay viewer
- priority % 2 简化规则和官方规则的偏差
```

第一版按 `docs/sim/rules.md` 里已确认的简化规则实现，然后用 replay 重放误差逐步修正规则。

## 非目标

第一阶段不做：

```text
- JAX/vectorized env
- PPO/self-play 训练框架
- 线上 websocket adapter
- 面向用户发布的完整 replay viewer
- 非 vanilla modifier
```

这些都应该等模拟器规则和 replay 重放稳定后再加。
