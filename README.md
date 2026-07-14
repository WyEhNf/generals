# Generals.io Adaptive Chunk Temporal Agent

This project is built upon [pigstd/generals.io-bot](https://github.com/pigstd/generals.io-bot),
adopting its local simulator, behavior cloning pre-training, PPO self-play, staged reward
shaping, and other foundational architecture and training configurations.

Building on the excellent work of the original project, this repository focuses on the
Limitations identified in its conclusion:

- **Temporal Memory & Belief State**: The original model makes per-frame decisions independently,
  lacking long-term tracking of enemy state under fog of war. We introduce a within-chunk gated
  accumulator and a causal Chunk Transformer, providing the policy network with a learnable
  compressed history representation and enemy occupancy belief — while preserving per-tick
  tactical decision frequency.
- **Adaptive Temporal Chunking**: Replace fixed K-step chunking with learnable boundary
  prediction, allowing the model to decide when to end a chunk based on current observations
  and belief state, enabling more flexible allocation of computation and memory resources.
- **Opponent Pool & Self-Play Upgrade (League / PFSP)**: Extend the original frozen opponent
  pool with historical snapshots, head-to-head matrix tracking, and Prioritized Fictitious
  Self-Play (PFSP) sampling to mitigate policy cycling and catastrophic forgetting.
- **Reward Attribution & Long-Range Credit Assignment**: Fix cross-rollout reward attribution
  consistency issues, and maintain continuous burn-in reconstruction of temporal memory in
  sequence PPO, improving the density of high-level training signals and long-range credit
  assignment efficiency.
- **High-Level Options & Router Integration**: Build high-level options (expansion, troop
  gathering, scouting, city assault, defensive positioning) atop Adaptive Chunk boundaries,
  and train option policies and a router with SMDP returns.

## Acknowledgments

Sincere gratitude and respect to [pigstd](https://github.com/pigstd) for the outstanding
ideas and solid engineering practice. The original `generals.io-bot` repository provides
a complete training pipeline, checkpoint system, and evaluation toolkit, upon which
follow-up research can be built on a reliable foundation.

## License

This project continues under the original [Apache License 2.0](LICENSE).

Original copyright notice:

```
Staged Reward Generals.io Bot
Copyright 2026 Zemu Zhu
```

This project includes a Python port of the Flobot benchmark bot and uses public
Generals.io replay data for behavior-cloning experiments. See
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for details.

## Related Links

- Original repository: [https://github.com/pigstd/generals.io-bot](https://github.com/pigstd/generals.io-bot)
- Original project report: [docs/report.pdf](docs/report.pdf)
