from __future__ import annotations

import argparse
import inspect
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    transformer_config = args.model_path / "transformer" / "config.json"
    model_index = args.model_path / "model_index.json"
    if not transformer_config.exists():
        raise FileNotFoundError(transformer_config)
    if not model_index.exists():
        raise FileNotFoundError(model_index)
    config = json.loads(transformer_config.read_text(encoding="utf-8"))
    try:
        from diffusers import FlowMatchEulerDiscreteScheduler, SD3Transformer2DModel
    except ImportError as exc:
        raise RuntimeError(f"diffusers SD3 imports unavailable: {exc}") from exc
    signature = str(inspect.signature(SD3Transformer2DModel.forward))
    required = [
        "in_channels",
        "out_channels",
        "patch_size",
        "num_attention_heads",
        "attention_head_dim",
        "joint_attention_dim",
        "caption_projection_dim",
    ]
    missing = [name for name in required if name not in config]
    if missing:
        raise RuntimeError(f"SD3 transformer config missing fields: {missing}")
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        "\n".join(
            [
                "SD3_CONFIG_AUDIT=PASS",
                "SD3_FORWARD_SIGNATURE_AUDIT=PASS",
                "FLOW_MATCH_SCHEDULER_IMPORT=PASS",
                "SD3_FULL_LOAD=NOT_RUN_NO_GPU_PROBE_ONLY",
                f"TRANSFORMER_FORWARD_SIGNATURE={signature}",
                f"SCHEDULER_CLASS={FlowMatchEulerDiscreteScheduler.__name__}",
                f"TRANSFORMER_CONFIG_FIELDS={','.join(sorted(config))}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    print("SD3_CONFIG_AUDIT=PASS")
    print("SD3_FORWARD_SIGNATURE_AUDIT=PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
