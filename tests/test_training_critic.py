from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

try:
    import torch

    TORCH_AVAILABLE = True
except ModuleNotFoundError:
    TORCH_AVAILABLE = False


@unittest.skipUnless(TORCH_AVAILABLE, "torch is not installed")
class CriticTrainingTest(unittest.TestCase):
    """Tests for outcome-target value-head training."""

    def test_train_value_head_smoke_writes_checkpoint_and_preserves_policy(self) -> None:
        """A tiny critic run should save a checkpoint without changing policy heads."""
        from generals_bot.training.critic import CriticConfig, train_value_head
        from generals_bot.training.encoding import ObservationEncoder
        from generals_bot.training.model import BCPolicyNet

        encoder = ObservationEncoder(history_length=1)
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            init_path = temp_path / "init.pt"
            output_path = temp_path / "critic.pt"
            _write_init_checkpoint(init_path, encoder, BCPolicyNet)
            before = torch.load(init_path, map_location="cpu")["model_state"]

            summary = train_value_head(
                CriticConfig(
                    checkpoint_path=str(init_path),
                    output_path=str(output_path),
                    updates=1,
                    num_envs=1,
                    episodes_per_update=1,
                    eval_episodes=1,
                    sample_stride=1,
                    train_epochs=1,
                    minibatch_size=16,
                    map_size=5,
                    max_turns=8,
                    device="cpu",
                    reset_value_head=True,
                )
            )

            checkpoint = torch.load(output_path, map_location="cpu")
            self.assertEqual(checkpoint["algo"], "value_head_outcome")
            self.assertEqual(checkpoint["global_update"], 1)
            self.assertIn("train_after/explained_variance", summary)
            self.assertIn("eval/explained_variance", summary)
            self.assertTrue(output_path.with_suffix(".pt.json").exists())
            after = checkpoint["model_state"]
            self.assertTrue(torch.equal(before["move_head.weight"], after["move_head.weight"]))
            self.assertTrue(torch.equal(before["pass_head.4.weight"], after["pass_head.4.weight"]))

    def test_evaluate_value_batch_reports_explained_variance(self) -> None:
        """Value batch evaluation should expose loss and explained variance."""
        from generals_bot.training.critic import OutcomeBatch, evaluate_value_batch
        from generals_bot.training.encoding import ObservationEncoder
        from generals_bot.training.model import BCPolicyNet

        encoder = ObservationEncoder(history_length=1)
        model = BCPolicyNet(encoder.num_channels, base_channels=8)
        batch = OutcomeBatch(
            x=torch.zeros((2, encoder.num_channels, 5, 5), dtype=torch.float16),
            target=torch.tensor([1.0, -1.0]),
            metadata=[{"env_id": 0, "player_id": 0, "turn": 0}],
        )

        metrics = evaluate_value_batch(model, batch, torch.device("cpu"), prefix="eval")

        self.assertIn("eval/value_loss", metrics)
        self.assertIn("eval/explained_variance", metrics)


def _write_init_checkpoint(path: Path, encoder, model_cls) -> None:
    """Write a tiny checkpoint usable by critic training tests."""
    model = model_cls(encoder.num_channels, base_channels=8)
    torch.save(
        {
            "model_state": model.state_dict(),
            "input_channels": encoder.num_channels,
            "feature_names": list(encoder.channel_names),
            "history_length": encoder.history_length,
            "base_channels": 8,
            "policy_mode": "hierarchical",
        },
        path,
    )


if __name__ == "__main__":
    unittest.main()
