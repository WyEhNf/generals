# 地图生成

## 目标

本地自博弈需要一个足够接近 1v1 vanilla Generals.io 的地图生成器。第一版目标不是复刻官方私有算法，而是生成：

```text
- 尺寸接近原版 1v1
- 将军距离合理
- mountain / city 密度合理
- 所有非 mountain 格子连通
- city army 在原版区间
- spawn / city 不明显偏向某一方
```

Replay 训练仍然优先使用 replay 自带地图。

## 公开依据

目前没有找到官方公开的完整地图生成算法。当前实现只使用能公开确认或间接确认的信息：

```text
- 本地 Hugging Face replay 数据显示当前 1v1 地图主要在 17-23 之间，18x18 最常见。
- 同一 guide 还提到 spawn 不是均匀分布，边角和较近距离更常见。
- generals-bots 的公开文档提到它有 generalsio map mode，目标是接近官方尺寸、city fairness 和 mountain 期望数量。
- generals-bots 文档还说明官方 city neutral army 是 40-50。
- 官方版本记录显示 2025-2026 年多次调整 spawn fairness / city fairness，说明真实规则包含公平性过滤，但没有公开具体算法。
```

来源：

```text
https://wiki.generals.io/1v1guide.html
https://pypi.org/project/generals-bots/
https://generals.io/versions
```

## 当前规则

第一版生成器输出 `GameConfig`，由模拟器直接消费。

默认参数：

```text
width:  weighted sample from [17, 23], strongly biased to 18
height: weighted sample from [17, 23], strongly biased to 18
players: 2
general min distance: Manhattan >= 15
mountain density: uniform [0.16, 0.24]
city density: uniform [0.025, 0.05]
city army: uniform integer [40, 50]
general initial army: 1
```

当前尺寸权重来自本地 raw replay 数据的宽高边际分布近似。完整数据中 18803 局的高频尺寸：

```text
18x18: 8000
19x18: 1264
18x19: 1247
17x18: 666
18x17: 646
```

General 采样：

```text
1. 第一个 general 在全图采样，但用 edge/corner bias 提高边角概率。
2. 第二个 general 只从曼哈顿距离 >= 15 的格子采样。
3. 第二个 general 同样有 edge/corner bias，并额外偏向“刚超过最小距离”的位置。
```

这个偏置来自 wiki 的经验描述：边角 spawn 和距离接近 15 的局面更常见。

Mountain 采样：

```text
1. 按本局采样到的 mountain density 生成 mountain 数量。
2. 不允许覆盖 general。
3. 生成后检查所有非 mountain 格子是否连通。
4. 生成后检查两个 general 周围至少有 2 个非 mountain 邻居。
5. 不满足则整张地图重采样。
```

City 采样：

```text
1. city 从非 mountain、非 general 格子中采样。
2. 不放在 general 曼哈顿距离 <= 2 的范围内。
3. city army 在 [40, 50]。
4. 做简单 city fairness 过滤：两个 general 附近半径 10 内的 city 数量差不能太大，army 总量差不能太大。
5. 不满足则整张地图重采样。
```

连通性：

```text
当前把 city 视为可通行，只有 mountain 阻塞。
要求所有非 mountain 格子属于同一个连通分量。
```

这样比“只要求两个 general 连通”更严格，方便训练和调试，也能避免 replay 可视化里出现永远不可达的小区域。

## 可调参数

代码里的 `MapGenerationConfig` 维护全部可调参数：

```python
MapGenerationConfig(
    min_size=17,
    max_size=23,
    size_weights=((17, 0.04), (18, 0.60), (19, 0.15), (20, 0.08), (21, 0.06), (22, 0.05), (23, 0.02)),
    min_general_distance=15,
    mountain_density_range=(0.16, 0.24),
    city_density_range=(0.025, 0.05),
    city_army_range=(40, 50),
    min_spawn_open_neighbors=2,
    local_fairness_radius=8,
    city_fairness_radius=10,
    max_attempts=1000,
)
```

后续如果用 replay 数据统计出真实分布，优先调整这个配置，而不是改模拟器规则。

## 已知疑点

```text
1. 官方真实 mountain density / city density 没有公开。
2. 官方 spawn fairness / city fairness 的具体评分函数没有公开。
3. 官方是否要求所有非 mountain 格子连通尚未确认。
4. city 是否可能离 general 很近、是否有额外禁区尚未确认。
5. 现在的 edge/corner bias 只是根据公开经验描述做的近似。
6. 尺寸权重来自当前 replay 数据的边际分布近似，不是官方生成算法。
```

这些需要以后用下载的 1v1 replay 地图统计来校准。
