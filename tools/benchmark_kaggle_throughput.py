from __future__ import annotations

import argparse
import time
import sys
from pathlib import Path
from typing import Any, Dict

import torch
import torch.nn.functional as F

try:
    from model.variant_c import VariantCConfig, VariantCModel
    from model.variant_e import VariantEConfig, VariantEModel
except ModuleNotFoundError:
    ROOT = Path(__file__).resolve().parents[1]
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from model.variant_c import VariantCConfig, VariantCModel
    from model.variant_e import VariantEConfig, VariantEModel


def _resolve_divisible_heads(width: int, requested_heads: int) -> int:
    heads = max(1, min(int(requested_heads), int(width)))
    while heads > 1 and (int(width) % heads) != 0:
        heads -= 1
    return max(1, heads)


def _variant_backend_status(model: torch.nn.Module) -> Dict[str, bool]:
    status = {
        "gdn_using_fallback": False,
    }
    for module in model.modules():
        if module.__class__.__name__ != "GatedDeltaNetBlock":
            continue
        if bool(getattr(module, "using_fallback", False)):
            status["gdn_using_fallback"] = True
    return status


def _build_model(args: argparse.Namespace) -> torch.nn.Module:
    if str(args.variant).lower() == "c":
        heads = _resolve_divisible_heads(int(args.d_model), int(args.num_attention_heads))
        return VariantCModel(
            VariantCConfig(
                vocab_size=int(args.vocab_size),
                d_model=int(args.d_model),
                n_layers=int(args.n_layers),
                max_sequence_length=int(args.seq_len),
                num_attention_heads=int(heads),
                ffn_expansion=int(max(1, args.ffn_expansion)),
                dropout=float(args.dropout),
                attention_dropout=float(args.attention_dropout),
            )
        )

    if str(args.variant).lower() == "e":
        gdn_inner_dim = max(128, int(round(float(args.d_model) * float(args.gdn_inner_ratio))))
        gdn_heads = _resolve_divisible_heads(gdn_inner_dim, int(args.gdn_num_heads))
        gqa_heads = _resolve_divisible_heads(int(args.d_model), int(args.gqa_num_heads))
        gqa_groups = max(1, min(int(args.gqa_groups), int(gqa_heads)))
        while gqa_groups > 1 and (gqa_heads % gqa_groups) != 0:
            gqa_groups -= 1

        return VariantEModel(
            VariantEConfig(
                vocab_size=int(args.vocab_size),
                d_model=int(args.d_model),
                n_layers=int(args.n_layers),
                max_sequence_length=int(args.seq_len),
                gdn_inner_dim=int(gdn_inner_dim),
                gdn_num_heads=int(gdn_heads),
                gqa_num_heads=int(gqa_heads),
                gqa_groups=int(gqa_groups),
                attention_every_n_layers=int(max(1, args.attention_every_n_layers)),
                dropout=float(args.dropout),
                attention_dropout=float(args.attention_dropout),
            )
        )

    raise ValueError("--variant must be 'c' or 'e'")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Quick throughput benchmark for Variant C/E on current Kaggle runtime."
    )
    parser.add_argument("--variant", type=str, choices=["c", "e"], required=True)
    parser.add_argument("--d_model", type=int, required=True)
    parser.add_argument("--n_layers", type=int, required=True)
    parser.add_argument("--seq_len", type=int, default=2048)
    parser.add_argument("--vocab_size", type=int, default=171)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--warmup_steps", type=int, default=3)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--attention_dropout", type=float, default=0.1)
    parser.add_argument("--num_attention_heads", type=int, default=8)
    parser.add_argument("--ffn_expansion", type=int, default=4)
    parser.add_argument("--attention_every_n_layers", type=int, default=2)
    parser.add_argument("--gdn_inner_ratio", type=float, default=0.5)
    parser.add_argument("--gdn_num_heads", type=int, default=4)
    parser.add_argument("--gqa_num_heads", type=int, default=8)
    parser.add_argument("--gqa_groups", type=int, default=4)
    parser.add_argument("--use_data_parallel", action="store_true")
    parser.add_argument("--allow_gdn_data_parallel", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    torch.manual_seed(int(args.seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(args.seed))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = _build_model(args)

    backend = _variant_backend_status(model)
    gpu_count = torch.cuda.device_count() if device.type == "cuda" else 1
    use_dp = bool(args.use_data_parallel and device.type == "cuda" and gpu_count > 1)

    if (
        use_dp
        and str(args.variant).lower() == "e"
        and not backend.get("gdn_using_fallback", False)
        and not bool(args.allow_gdn_data_parallel)
    ):
        use_dp = False
        print(
            "[warning] Disabled DataParallel for Variant E with real GDN kernels "
            "(stability guard). Pass --allow_gdn_data_parallel to override."
        )

    model = model.to(device)
    if use_dp:
        model = torch.nn.DataParallel(model, device_ids=list(range(gpu_count)))

    optimizer = torch.optim.AdamW(model.parameters(), lr=float(args.learning_rate))

    total_steps = int(max(1, args.steps))
    warmup_steps = int(max(0, args.warmup_steps))

    measured_elapsed = 0.0
    measured_tokens = 0

    bsz = int(max(1, args.batch_size))
    seq_len = int(max(8, args.seq_len))
    vocab = int(max(16, args.vocab_size))

    for step in range(total_steps + warmup_steps):
        token_ids = torch.randint(0, vocab, (bsz, seq_len), dtype=torch.long, device=device)
        onset_times = (
            torch.arange(seq_len, dtype=torch.float32, device=device)
            .unsqueeze(0)
            .repeat(bsz, 1)
            * 0.1
        )
        targets = torch.randint(0, vocab, (bsz, seq_len), dtype=torch.long, device=device)

        t0 = time.time()
        optimizer.zero_grad(set_to_none=True)
        logits = model(token_ids=token_ids, onset_times=onset_times)
        if isinstance(logits, tuple):
            logits = logits[0]

        loss = F.cross_entropy(
            logits[:, :-1, :].reshape(-1, logits.shape[-1]),
            targets[:, 1:].reshape(-1),
        )
        loss.backward()
        optimizer.step()
        if device.type == "cuda":
            torch.cuda.synchronize()
        dt = time.time() - t0

        if step >= warmup_steps:
            measured_elapsed += dt
            measured_tokens += bsz * seq_len

    tokens_per_sec = measured_tokens / max(1e-6, measured_elapsed)
    steps_per_sec = total_steps / max(1e-6, measured_elapsed)

    print("Throughput benchmark")
    print(f"  variant: {args.variant}")
    print(f"  d_model: {args.d_model} | n_layers: {args.n_layers} | seq_len: {seq_len}")
    print(f"  batch_size: {bsz} | steps: {total_steps} (+{warmup_steps} warmup)")
    print(f"  device: {device} | gpus: {gpu_count} | data_parallel: {use_dp}")
    print(f"  gdn_fallback: {backend.get('gdn_using_fallback', False)}")
    print(f"  mean_step_seconds: {measured_elapsed / max(1, total_steps):.4f}")
    print(f"  steps_per_sec: {steps_per_sec:.4f}")
    print(f"  tokens_per_sec: {tokens_per_sec:.2f}")


if __name__ == "__main__":
    main()
