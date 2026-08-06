from __future__ import annotations

import torch

from sd3_rgda.pilot_engine import MockPilotRunner


def test_real_resume_equivalence(tmp_path) -> None:
    continuous = MockPilotRunner(tmp_path / "continuous")
    continuous_result = continuous.run(10)
    split = MockPilotRunner(tmp_path / "split")
    first = split.run(5)
    resumed = MockPilotRunner(tmp_path / "resumed")
    resumed_result = resumed.run(10, resume_from=first.checkpoint_path)
    assert resumed_result.losses == continuous_result.losses
    assert resumed_result.sample_usage_counts == continuous_result.sample_usage_counts
    assert resumed_result.next_position == continuous_result.next_position
    continuous_state = torch.load(continuous_result.checkpoint_path, map_location="cpu", weights_only=False)
    resumed_state = torch.load(resumed_result.checkpoint_path, map_location="cpu", weights_only=False)
    for key, tensor in continuous_state["modules"]["normal_adapter"].items():
        assert torch.equal(tensor, resumed_state["modules"]["normal_adapter"][key])
    assert resumed_state["optimizer_state"]["state"].keys() == continuous_state["optimizer_state"]["state"].keys()
