# 模拟器接口设计

## 目标

模拟器第一阶段只支持 1v1 vanilla Generals.io。核心目标是让以下三类场景共用同一套策略接口：

```text
1. replay 重放生成监督学习样本
2. 本地 agent vs agent 自博弈和评测
3. 线上 adapter 把真实服务器 update 转成同格式 observation
```

模拟器是规则层，不应该知道 websocket、训练框架、模型结构，也不应该依赖某个具体 bot。

## 环境接口

高层同步接口用于本地自博弈：

```python
env = GeneralsEnv(config)
obs = env.reset(seed=0)

while not env.done:
    actions = {
        player_id: policy.act(obs[player_id])
        for player_id in env.alive_players
    }
    obs, rewards, done, info = env.step(actions)
```

低层接口用于 replay 重放和线上语义对齐：

```python
env.submit(player_id, action)
env.tick()
obs = env.observe(player_id)
```

`step(actions)` 是低层接口的封装。推荐语义：

```text
1. 把本 tick 新收到的 actions 放入动作队列
2. 执行每个玩家队列里已经到期的一个合法动作
3. 推进自然增长和胜负检查
4. 返回新 observation
```

关键要求：本地 vanilla 模拟器不实现固定网络延迟。`obs_t -> action_t` 会在本 tick 入队并参与结算，结果最早在 `obs_{t+1}` 可见。

线上 adapter 后续可以在模拟器外层模拟网络 RTT / tick 相位造成的额外延迟。根据当前实验，`game_speed = 1` 基本不受影响；`game_speed = 2/4` 可能因为 tick 间隔短而表现出额外 update 延迟。

## Bot 接口

Bot 只需要实现策略接口：

```python
class Policy(Protocol):
    def reset(self, player_id: int, config: GameConfig) -> None:
        ...

    def act(self, observation: Observation) -> Action | list[Action] | PassAction:
        ...
```

`act()` 允许返回动作列表。原因是 Generals.io 服务器本来支持队列：启发式 bot 可以在一次决策中规划一整条路径，并把多个 `attack` 连续提交给服务器。

本地模拟器和线上 adapter 都消费同一种返回值：

```text
Action        -> 提交一个动作
list[Action]  -> 按顺序提交多个动作
PassAction    -> 本 tick 不提交动作
```

## Action

核心动作格式使用 tile index，而不是 row/col/direction。row/col/direction 只作为工具函数转换。

```python
@dataclass(frozen=True)
class Action:
    player_id: int
    start: int
    end: int
    split: bool = False
```

辅助类型：

```python
@dataclass(frozen=True)
class PassAction:
    player_id: int

@dataclass(frozen=True)
class DirectionMove:
    player_id: int
    row: int
    col: int
    direction: Direction
    split: bool = False
```

动作合法性由引擎在执行时判断。非法动作不抛异常，直接跳过，模拟线上服务器行为。

## Observation

Observation 是 bot、监督学习、线上 adapter 的统一输入。它是玩家视角，不是全局真值。

建议字段：

```python
@dataclass(frozen=True)
class Observation:
    player_id: int
    turn: int
    width: int
    height: int

    visible: np.ndarray          # bool[H, W]
    explored: np.ndarray         # bool[H, W]

    armies: np.ndarray           # int[H, W], 不可见处为 0
    owner: np.ndarray            # int[H, W], own/enemy/neutral/fog/mountain/fog_obstacle

    own_tiles: np.ndarray        # bool[H, W]
    enemy_tiles: np.ndarray      # bool[H, W]
    neutral_tiles: np.ndarray    # bool[H, W]
    mountains: np.ndarray        # bool[H, W], 仅可见山
    cities: np.ndarray           # bool[H, W], 仅可见城市
    known_cities: np.ndarray     # bool[H, W], 曾经见过的城市
    generals: np.ndarray         # bool[H, W], 仅可见将军
    known_generals: np.ndarray   # bool[H, W], 曾经见过且仍存活的将军
    known_enemy_generals: np.ndarray # bool[H, W], 曾经见过且仍存活的敌方将军
    fog: np.ndarray              # bool[H, W]
    fog_obstacles: np.ndarray    # bool[H, W], fog 中已知有 city/mountain 的格子

    own_army: int                # UI 公开信息：己方总兵力
    own_land: int                # UI 公开信息：己方总土地
    enemy_army: int              # UI 公开信息：敌方总兵力
    enemy_land: int              # UI 公开信息：敌方总土地
    last_moves: list[ObservedMove]
    priority: int
```

`generals` 只表示当前视野内的将军。`known_generals` / `known_enemy_generals`
来自玩家记忆：一旦某个将军被看见，后续即使该格子重新进入 fog，也会继续保留
这个已知位置，直到对应玩家死亡。这个字段用于模拟人类玩家会记住敌方王位的事实，
也让 BC / self-play / 线上 adapter 共用同一种输入语义。

`cities` 只表示当前视野内的城市。`known_cities` 来自玩家记忆，用于接入 Flobot
这类依赖 Generals.io `cities_diff` 语义的 benchmark bot。训练 encoder 当前不使用
`known_cities`，所以已有 checkpoint 的输入通道不变。

`own_*` / `enemy_*` 是 1v1 策略和训练使用的显式公开信息。Observation 不保留通用 `scores` 列表，避免为了未来多人扩展提前增加不需要的接口复杂度。

训练时可以把 Observation 编码成 tensor，但模拟器本身先保留语义清晰的数据结构。

## Replay 接口

Replay loader 负责读取外部数据格式，转换成内部结构：

```python
@dataclass(frozen=True)
class Replay:
    replay_id: str
    width: int
    height: int
    generals: list[int]
    mountains: list[int]
    cities: list[int]
    city_armies: list[int]
    moves: list[ReplayMove]

@dataclass(frozen=True)
class ReplayMove:
    player_id: int
    start: int
    end: int
    split: bool
    turn: int
```

Replay replayer 输出监督学习样本：

```python
for sample in replay_to_samples(replay):
    observation = sample.observation
    label = sample.action
```

如果 replay move 的 turn 语义和模拟器 tick 语义有偏差，需要通过 golden test 校正。
