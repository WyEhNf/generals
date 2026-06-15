# 游戏规则细节

## 范围

第一阶段只实现 1v1 vanilla Generals.io，不支持：

```text
swamp
desert
torus
defection
chess clock
teams / FFA
lookout / observatory 等新版特殊建筑
```

## Tick 和 Turn

模拟器内部采用 half-turn 作为 tick。官方 UI 的 1 turn 通常对应两个 bot update / half-turn。

当前本地 vanilla 模拟器不引入额外网络延迟：玩家基于 `obs_t` 提交的 `action_t` 会在本 tick 结算，结果可以在 `obs_{t+1}` 中看到。

线上 bot API 仍可能表现出额外延迟，尤其是在 `game_speed > 1` 时。我们后续实测发现：

```text
- game_speed = 1: 通常 action_t 在下一个 update 可见
- game_speed = 2: 有时下一个 update 可见，有时多等一个 update
- game_speed = 4: 常见多等 1-2 个 update
```

这更像是网络 RTT、客户端计算时间和服务器 tick 相位造成的线上 adapter 问题，不应写成本地模拟器的固定游戏规则。第一阶段本地训练和自博弈按 `game_speed = 1` 语义处理。

因此每个 tick 的顺序固定为：

```text
1. 接收本 tick 新提交动作，放入玩家队列尾部
2. 从每个玩家的队列中取一个可执行动作
3. 按 priority % 2 决定执行顺序并执行动作
4. 检查 general capture / 胜负
5. 应用自然增长
6. 更新 player memory
7. turn += 1
```

对应到 `step(actions)`：

```text
obs_t -> step(action_t) -> obs_{t+1}
action_t 在 step(action_t) 的开头入队
本 tick 开始结算时可以执行每个玩家队首的一个合法动作
所以 action_t 的结果最早出现在 obs_{t+1}
```

## Priority

第一版不复刻官方可能存在的复杂 priority 规则，只用简单交替先手：

```text
first_player = priority % 2
second_player = 1 - first_player
```

默认 `priority = turn`，所以：

```text
偶数 tick: player 0 先执行
奇数 tick: player 1 先执行
```

同 tick 双方互攻、抢城、抢 general 都按这个顺序顺次结算，不做同步结算。

## 动作队列

每个玩家有动作队列，用来模拟连续提交路径：

```python
class ActionQueue:
    pending: deque[QueuedAction]
```

规则：

```text
- step(actions) 开头先 submit，再执行本 tick 队首动作
- env.submit(...) 只入队；随后 env.tick() 会执行到期队首动作
- 每 tick 最多尝试执行每个玩家队首的一个 action
- 队首非法则丢弃，并继续尝试下一个，直到执行一个合法动作或队列为空
- list[Action] 按顺序连续入队
```

非法动作：

```text
- start/end 越界
- start/end 不相邻
- start 不是该玩家领地
- start 兵力 <= 1
- end 是 mountain
```

非法动作跳过是为了匹配线上队列行为。长路径中前面某一步失败，后续动作可能自然变非法，服务器应该跳过这些动作，而不是让 bot 崩溃。

## 攻击和移动

普通移动：

```text
army_to_move = source_army - 1
source 留 1
```

Split move：

```text
army_to_move = floor(source_army / 2)
source 至少留 1
```

目标是己方格：

```text
target_army += army_to_move
source_army -= army_to_move
```

目标是中立或敌方格：

```text
if army_to_move > target_army:
    target_owner = attacker
    target_army = army_to_move - target_army
else:
    target_army = target_army - army_to_move
```

## General Capture

当玩家的 general 被敌方占领：

```text
- 被占领方死亡
- 游戏结束，攻击方胜利
```

第一版不实现死亡后的完整领地转移，因为 1v1 训练和评测只需要胜负。以后如果要做 replay viewer 或多人局，再补全领地转移细节。

## 自然增长

初始状态：

```text
- 每个 general 初始 army = 1
- city 使用地图配置里的中立 army；如果未指定，默认 40
- 普通 neutral land 初始 army = 0
```

先执行动作，再自然增长。增长按 half-turn 规则实现：

```text
- general 和 city 每 2 half-turn 增长 1
- 所有 owned land 每 50 half-turn 增长 1
```

默认增长发生在动作和胜负检查之后：

```text
enqueue submitted actions
execute queued actions
check winner
apply growth
update memory
turn += 1
```

增长判断使用 tick 结束后的 `next_turn = turn + 1`：

```text
if next_turn % 2 == 0:
    generals/cities +1

if next_turn % 50 == 0:
    all owned land +1
```

因此初始 `turn=0` 不会立刻增长。

## 视野

Vanilla 视野：

```text
own tile 周围 3x3 可见
不可见普通格显示 fog
不可见 city/mountain 显示 fog_obstacle
可见敌方 tile 显示 owner 和 army
不可见 army 不泄露
```

Player memory：

```text
- explored：曾经可见过的格子
- seen_cities：曾经见过的城市位置
- seen_generals：曾经见过的将军位置
- seen_obstacles：曾经见过或 fog 中显示为 obstacle 的位置
```

监督学习时，Observation 必须只包含该玩家当时能知道的信息，不能直接使用 replay 的全图真值。

## 地图生成

Replay 训练直接使用 replay 自带地图，不需要生成地图。本地自博弈地图生成器的具体参数、连通性和公平性过滤写在 [地图生成](map-generation.md) 中。

## 规则疑点

以下规则先按上文版本实现，但后续需要用真实 replay 或线上日志做 golden test：

```text
1. Replay move.turn 的语义
   需要确认 replay 里的 turn 是客户端 submit tick，还是服务器 execute tick。

2. 动作延迟
   当前本地模拟器不实现固定延迟，按 action_t 最早在 obs_{t+1} 可见处理。
   线上 speed=1 基本符合这个行为；speed=2/4 的额外可见延迟很可能来自网络 RTT 和服务器 tick 相位。
   线上 adapter 以后需要单独处理高 speed 下的异步问题，但不应污染本地规则层。

3. 自然增长边界
   当前假设 turn=0 不增长，并按 next_turn 判断增长 tick；仍需确认是否和官方完全一致。

4. City 当 tick 占领后的增长
   当前按“动作后增长”处理，所以新占领 city 如果正好遇到增长 tick，会立即增长。这点需要验证。

5. General capture 后领地转移
   第一版直接结束游戏，不模拟死后领地转移。若后续需要精确 replay viewer，需要补齐。

6. Priority
   第一版使用 priority % 2 交替先手。官方可能有更复杂的同 tick 冲突规则，暂不实现。
```
