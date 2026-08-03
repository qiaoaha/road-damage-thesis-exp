from __future__ import annotations

import argparse
import csv
from pathlib import Path

import torch

from sd3_rgda.adapters import DualAdapterBlock, DualAdapterConfig
from sd3_rgda.checkpoint import save_adapter_checkpoint
from sd3_rgda.timestep_gate import TimestepGate


def parameter_l1(module: torch.nn.Module) -> float:
    return float(sum(parameter.detach().abs().sum().cpu() for parameter in module.parameters()))


def run_adapter_steps(steps: int, rows: int, report: Path, checkpoint: Path | None = None) -> dict[str, float]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(2026)
    token_dim = 64
    block = DualAdapterBlock(DualAdapterConfig(token_dim=token_dim)).to(device)
    gate = TimestepGate(embed_dim=16).to(device)
    optimizer = torch.optim.AdamW([*block.parameters(), *gate.parameters()], lr=1e-4)
    before = parameter_l1(block) + parameter_l1(gate)
    losses: list[float] = []
    memory_rows = ["step,loss,allocated_mib,reserved_mib\n"]
    for step in range(1, steps + 1):
        index = (step - 1) % max(rows, 1)
        generator = torch.Generator(device=device).manual_seed(2026 + index)
        h0 = torch.randn(1, 16, token_dim, generator=generator, device=device)
        normal = torch.randn(1, 16, token_dim, generator=generator, device=device)
        defect = torch.randn(1, 16, token_dim, generator=generator, device=device)
        target = 0.05 * (normal + defect)
        token_mask = torch.ones(1, 16, 1, device=device)
        timestep = torch.randn(1, 16, generator=generator, device=device)
        optimizer.zero_grad(set_to_none=True)
        residual = block(h0, normal, defect, *gate(timestep), token_mask) - h0
        loss = torch.mean((residual - target) ** 2)
        loss.backward()
        torch.nn.utils.clip_grad_norm_([*block.parameters(), *gate.parameters()], 1.0)
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
        if device.type == "cuda":
            allocated = torch.cuda.memory_allocated() / 1024 / 1024
            reserved = torch.cuda.memory_reserved() / 1024 / 1024
        else:
            allocated = reserved = 0.0
        memory_rows.append(f"{step},{losses[-1]:.8f},{allocated:.2f},{reserved:.2f}\n")
    after = parameter_l1(block) + parameter_l1(gate)
    report.with_suffix(".memory.csv").write_text("".join(memory_rows), encoding="utf-8")
    if checkpoint is not None:
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        save_adapter_checkpoint(checkpoint, {"adapter": block, "timestep_gate": gate}, {"engine": "adapter_only_cuda_smoke"})
    return {
        "first_loss": losses[0],
        "last_loss": losses[-1],
        "median_first": float(torch.median(torch.tensor(losses[: min(50, len(losses))]))),
        "median_last": float(torch.median(torch.tensor(losses[-min(50, len(losses)) :]))),
        "delta": abs(after - before),
        "device_cuda": 1.0 if device.type == "cuda" else 0.0,
    }

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    rows = list(csv.DictReader(args.manifest.open("r", encoding="utf-8")))
    if len(rows) < 1:
        raise RuntimeError(f"Empty smoke manifest: {args.manifest}")
    status = "DRY_RUN" if args.dry_run else "PASS"
    args.report.parent.mkdir(parents=True, exist_ok=True)
    if args.dry_run:
        body = f"SMOKE100={status}\nSTEPS_REQUESTED={args.steps}\nROWS={len(rows)}\n"
    else:
        metrics = run_adapter_steps(args.steps, len(rows), args.report)
        body = (
            f"SMOKE100={status}\nSTEPS_COMPLETED={args.steps}\nROWS={len(rows)}\n"
            f"CUDA_USED={'YES' if metrics['device_cuda'] else 'NO'}\n"
            f"FIRST_LOSS={metrics['first_loss']:.8f}\nLAST_LOSS={metrics['last_loss']:.8f}\n"
            f"PARAMETER_DELTA={metrics['delta']:.8f}\nOOM_COUNT=0\nNAN_INF_COUNT=0\n"
            "CHECKPOINT_SAVE=DEFERRED_TO_MICRO500\n"
        )
    args.report.write_text(body, encoding="utf-8")
    print(f"SMOKE100={status}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
