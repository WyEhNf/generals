from __future__ import annotations

import json
import math
import tempfile
import unittest
from pathlib import Path

import torch

from generals_bot.training.encoding import ObservationEncoder
from generals_bot.training.model import BCPolicyNet, POLICY_MODE


class RouterTrainingTest(unittest.TestCase):
    """Tests for frozen-expert router PPO plumbing."""

    def test_router_expert_config_rejects_duplicate_ids(self) -> None:
        """Router expert configs should reject ambiguous expert ids."""
        from generals_bot.training.router import load_router_expert_config

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            checkpoint = root / "model.pt"
            checkpoint.write_bytes(b"placeholder")
            config = root / "router.json"
            config.write_text(
                json.dumps(
                    {
                        "version": 1,
                        "experts": [
                            {"id": "same", "path": "model.pt"},
                            {"id": "same", "path": "model.pt"},
                        ],
                    }
                ),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "duplicate"):
                load_router_expert_config(config)

    def test_router_warm_start_zeroes_initial_expert_logits(self) -> None:
        """Router warm-start should initialize a uniform expert distribution."""
        from generals_bot.training.router import (
            RouterPPOConfig,
            _load_or_init_router_model,
            load_router_expert_config,
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = root / "source.pt"
            _write_policy_checkpoint(source)
            config_path = root / "router.json"
            config_path.write_text(
                json.dumps(
                    {
                        "version": 1,
                        "experts": [
                            {"id": "a", "path": "source.pt"},
                            {"id": "b", "path": "source.pt"},
                        ],
                    }
                ),
                encoding="utf-8",
            )
            expert_config = load_router_expert_config(config_path)

            model, encoder, _, _, _, _ = _load_or_init_router_model(
                RouterPPOConfig(
                    expert_config=str(config_path),
                    init_router_from=str(source),
                ),
                expert_config,
                torch.device("cpu"),
            )

            x = torch.zeros(3, encoder.num_channels, 5, 5)
            output = model(x)
            self.assertTrue(torch.allclose(output["expert_logits"], torch.zeros(3, 2)))

    def test_evaluate_router_experts_returns_categorical_stats(self) -> None:
        """Router log-probs and entropy should match a categorical policy."""
        from generals_bot.training.router import evaluate_router_experts_from_output

        output = {
            "expert_logits": torch.tensor([[0.0, 0.0], [2.0, 0.0]]),
            "value": torch.tensor([0.5, -0.25]),
        }
        labels = torch.tensor([1, 0])

        log_probs, entropy, values, max_prob = evaluate_router_experts_from_output(
            output,
            labels,
        )

        self.assertAlmostEqual(float(log_probs[0]), -math.log(2), places=6)
        self.assertGreater(float(log_probs[1]), -0.2)
        self.assertAlmostEqual(float(entropy[0]), math.log(2), places=6)
        self.assertTrue(torch.equal(values, output["value"]))
        self.assertAlmostEqual(float(max_prob[0]), 0.5, places=6)

    def test_pad_router_batch_marks_dummy_samples_zero_weight(self) -> None:
        """Router DDP padding should preserve shapes and ignore dummy rows."""
        from generals_bot.training.router import RouterRolloutBatch, _pad_router_batch

        batch = RouterRolloutBatch(
            x=torch.ones(2, 3, 4, 4),
            expert_label=torch.tensor([0, 1]),
            old_log_prob=torch.zeros(2),
            old_value=torch.zeros(2),
            rewards=torch.zeros(2),
            done=torch.zeros(2, dtype=torch.bool),
            returns=torch.zeros(2),
            advantages=torch.ones(2),
            sample_weight=torch.ones(2),
            stream_ids=[(0, 0), (0, 1)],
            metadata=[{"env_id": 0}, {"env_id": 0}],
        )

        padded = _pad_router_batch(batch, 4)

        self.assertEqual(padded.size, 4)
        self.assertEqual(padded.real_size, 2)
        self.assertTrue(torch.equal(padded.sample_weight, torch.tensor([1, 1, 0, 0], dtype=torch.float32)))
        self.assertEqual(padded.stream_ids[-1], (-1, -1))
        self.assertEqual(padded.metadata[-1]["padding"], 1)


def _write_policy_checkpoint(path: Path) -> None:
    """Write a tiny hierarchical policy checkpoint for router tests."""
    encoder = ObservationEncoder()
    model = BCPolicyNet(encoder.num_channels, base_channels=8)
    torch.save(
        {
            "model_state": model.state_dict(),
            "input_channels": encoder.num_channels,
            "feature_names": list(encoder.channel_names),
            "history_length": encoder.history_length,
            "base_channels": 8,
            "policy_mode": POLICY_MODE,
        },
        path,
    )


if __name__ == "__main__":
    unittest.main()
