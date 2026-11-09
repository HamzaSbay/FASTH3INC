"""Remote GGUF census tool for FastH3-8-Step-V2 and Qwen3-VL text encoder.

Inspects model headers remotely over HTTP range requests, computes byte counts
by component and layer, outputs exact statistics for NVMe streaming and AdaLN folding,
and saves docs/data/census.json.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from typing import Any

try:
    from . import gguf_remote
except ImportError:
    import os
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from tools import gguf_remote

DEFAULT_DIT_URL = (
    "https://huggingface.co/realrebelai/FastH3-V2_GGUFs/resolve/main/FastH3-8-Step-V2-Q4_K_M.gguf"
)
DEFAULT_TE_URL = (
    "https://huggingface.co/realrebelai/FastH3-V2_GGUFs/resolve/main/Text%20Encoders/qwen3vl-32B-MiniMax-H3-Q4_K_M.gguf"
)
DEFAULT_OUTPUT_PATH = "docs/data/census.json"


def analyze_dit(header: gguf_remote.GgufHeader, url: str) -> dict[str, Any]:
    """Analyze DiT GGUF tensors and group them into streaming components."""
    tensors = header.tensors
    total_bytes = sum(t.nbytes for t in tensors)

    # Detailed per-block & per-category categorization
    blocks: dict[int, list[gguf_remote.TensorInfo]] = {}
    run_once: list[gguf_remote.TensorInfo] = []

    for t in tensors:
        m = re.match(r"^blocks\.(\d+)\.", t.name)
        if m:
            b_idx = int(m.group(1))
            blocks.setdefault(b_idx, []).append(t)
        else:
            run_once.append(t)

    num_blocks = len(blocks)

    # AdaLN in blocks
    adaln_weight_tensors = [t for t in tensors if re.match(r"^blocks\.\d+\.adaln_proj\.linear\.weight", t.name)]
    adaln_bias_tensors = [t for t in tensors if re.match(r"^blocks\.\d+\.adaln_proj\.linear\.bias", t.name)]
    adaln_weight_bytes = sum(t.nbytes for t in adaln_weight_tensors)
    adaln_bias_bytes = sum(t.nbytes for t in adaln_bias_tensors)
    adaln_total_bytes = adaln_weight_bytes + adaln_bias_bytes

    # Per-block breakdown (using block 0 as canonical exemplar)
    b0_tensors = blocks.get(0, [])
    b0_adaln_bytes = sum(t.nbytes for t in b0_tensors if "adaln" in t.name)
    b0_trunk_bytes = sum(t.nbytes for t in b0_tensors if "adaln" not in t.name)
    b0_total_bytes = b0_adaln_bytes + b0_trunk_bytes

    # Trunk bytes (all blocks without AdaLN)
    trunk_tensors = [t for t in tensors if t.name.startswith("blocks.") and "adaln" not in t.name]
    trunk_bytes = sum(t.nbytes for t in trunk_tensors)

    # Run-once components breakdown
    run_once_bytes = sum(t.nbytes for t in run_once)
    run_once_categories: dict[str, int] = {}
    for t in run_once:
        cat = t.name.split(".")[0]
        run_once_categories[cat] = run_once_categories.get(cat, 0) + t.nbytes

    # Quantization types breakdown
    quant_types: dict[str, int] = {}
    for t in tensors:
        quant_types[t.type] = quant_types.get(t.type, 0) + t.nbytes

    # Canonical tensor groups
    groups: dict[str, dict[str, Any]] = {}
    for t in tensors:
        # Group name by pattern: blocks.N.<submodule> or non-block prefix
        m = re.match(r"^(blocks\.\d+)\.([^.]+)", t.name)
        if m:
            g_name = f"blocks.N.{m.group(2)}"
        else:
            g_name = t.name.split(".")[0]
        if g_name not in groups:
            groups[g_name] = {"bytes": 0, "count": 0, "types": set()}
        groups[g_name]["bytes"] += t.nbytes
        groups[g_name]["count"] += 1
        groups[g_name]["types"].add(t.type)

    for g_info in groups.values():
        g_info["types"] = sorted(list(g_info["types"]))

    # 2-slot ring buffer memory footprint (2 * trunk block size)
    ring_2slot_bytes = 2 * b0_trunk_bytes

    return {
        "url": url,
        "arch": header.kv.get("general.architecture", "unknown"),
        "total_tensors": len(tensors),
        "total_bytes": total_bytes,
        "num_blocks": num_blocks,
        "adaln": {
            "weight_bytes": adaln_weight_bytes,
            "bias_bytes": adaln_bias_bytes,
            "total_bytes": adaln_total_bytes,
            "tensor_count": len(adaln_weight_tensors) + len(adaln_bias_tensors),
        },
        "trunk": {
            "total_bytes": trunk_bytes,
            "tensor_count": len(trunk_tensors),
            "per_block_bytes": b0_trunk_bytes,
            "max_block_run_bytes": b0_trunk_bytes,
            "ring_2slot_bytes": ring_2slot_bytes,
        },
        "per_block": {
            "total_bytes": b0_total_bytes,
            "adaln_bytes": b0_adaln_bytes,
            "trunk_bytes": b0_trunk_bytes,
        },
        "run_once": {
            "total_bytes": run_once_bytes,
            "tensor_count": len(run_once),
            "categories": run_once_categories,
        },
        "quant_types": quant_types,
        "groups": groups,
    }


def analyze_te(header: gguf_remote.GgufHeader, url: str) -> dict[str, Any]:
    """Analyze Qwen3-VL Text Encoder GGUF tensors."""
    tensors = header.tensors
    total_bytes = sum(t.nbytes for t in tensors)

    layers_lt50 = [t for t in tensors if any(t.name.startswith(f"model.layers.{i}.") for i in range(50))]
    layers_ge50 = [t for t in tensors if any(t.name.startswith(f"model.layers.{i}.") for i in range(50, 200))]
    embed_tokens = [t for t in tensors if t.name == "model.embed_tokens.weight"]
    visual_tensors = [t for t in tensors if t.name.startswith("visual.")]
    other_tensors = [
        t for t in tensors
        if not t.name.startswith("model.layers.")
        and not t.name.startswith("visual.")
        and t.name != "model.embed_tokens.weight"
    ]

    lt50_bytes = sum(t.nbytes for t in layers_lt50)
    ge50_bytes = sum(t.nbytes for t in layers_ge50)
    embed_bytes = sum(t.nbytes for t in embed_tokens)
    visual_bytes = sum(t.nbytes for t in visual_tensors)
    other_bytes = sum(t.nbytes for t in other_tensors)

    # Quantization types
    quant_types: dict[str, int] = {}
    for t in tensors:
        quant_types[t.type] = quant_types.get(t.type, 0) + t.nbytes

    return {
        "url": url,
        "arch": header.kv.get("general.architecture", "unknown"),
        "total_tensors": len(tensors),
        "total_bytes": total_bytes,
        "layers_lt50": {
            "bytes": lt50_bytes,
            "tensor_count": len(layers_lt50),
            "layer_count": len({int(re.search(r"model\.layers\.(\d+)\.", t.name).group(1)) for t in layers_lt50}),
        },
        "layers_ge50": {
            "bytes": ge50_bytes,
            "tensor_count": len(layers_ge50),
        },
        "embed_tokens": {
            "bytes": embed_bytes,
            "tensor_count": len(embed_tokens),
        },
        "visual": {
            "bytes": visual_bytes,
            "tensor_count": len(visual_tensors),
        },
        "other": {
            "bytes": other_bytes,
            "tensor_count": len(other_tensors),
        },
        "quant_types": quant_types,
    }


def compute_reductions_ledger(dit: dict[str, Any]) -> dict[str, Any]:
    """Compute exact bytes touched at each stage for the README ledger table."""
    total_dit = dit["total_bytes"]
    adaln_total = dit["adaln"]["total_bytes"]
    trunk_only = dit["trunk"]["total_bytes"]
    after_adaln_folded = total_dit - adaln_total
    ring_2slot = dit["trunk"]["ring_2slot_bytes"]

    return {
        "official_bf16_gb": 70.10,
        "q4_k_m_gguf": {
            "bytes": total_dit,
            "gb": round(total_dit / 1e9, 2),
            "gib": round(total_dit / (1024**3), 2),
        },
        "after_adaln_folded": {
            "bytes": after_adaln_folded,
            "gb": round(after_adaln_folded / 1e9, 2),
            "gib": round(after_adaln_folded / (1024**3), 2),
        },
        "trunk_per_step": {
            "bytes": trunk_only,
            "gb": round(trunk_only / 1e9, 2),
            "gib": round(trunk_only / (1024**3), 2),
        },
        "ram_floor": {
            "ring_2slot_bytes": ring_2slot,
            "ring_2slot_mb": round(ring_2slot / (1024**2), 2),
            "estimated_phase_gb": 1.5,
        },
    }


def run_census(dit_url: str, te_url: str, out_path: str = DEFAULT_OUTPUT_PATH) -> dict[str, Any]:
    """Execute full census against remote/local models and write docs/data/census.json."""
    print(f"[*] Fetching DiT header from: {dit_url}")
    dit_header = gguf_remote.header(dit_url)
    print(f"    Loaded {len(dit_header.tensors)} tensors, {len(dit_header.kv)} KV entries.")

    print(f"[*] Fetching Text Encoder header from: {te_url}")
    te_header = gguf_remote.header(te_url)
    print(f"    Loaded {len(te_header.tensors)} tensors, {len(te_header.kv)} KV entries.")

    dit_stats = analyze_dit(dit_header, dit_url)
    te_stats = analyze_te(te_header, te_url)
    ledger = compute_reductions_ledger(dit_stats)

    full_census = {
        "dit": dit_stats,
        "te": te_stats,
        "reductions_ledger": ledger,
    }

    # Ensure output directory exists
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(full_census, f, indent=2)
    print(f"[+] Census written to: {out_path}")

    # Print clean summary tables to stdout
    print_census_summary(dit_stats, te_stats, ledger)
    return full_census


def print_census_summary(dit: dict[str, Any], te: dict[str, Any], ledger: dict[str, Any]) -> None:
    """Print readable ASCII summary tables to stdout."""
    print("\n" + "=" * 78)
    print("FastH3-8-Step-V2 (DiT) Remote Census Summary")
    print("=" * 78)
    print(f"Total Tensors: {dit['total_tensors']}")
    print(f"Total Size:    {dit['total_bytes']:,} bytes ({dit['total_bytes']/1e9:.3f} GB)")
    print(f"Architecture:  {dit['arch']}")
    print(f"DiT Blocks:    {dit['num_blocks']} blocks")

    print("\n[DiT Component Breakdown]")
    print(f"  Trunk (50 blocks non-AdaLN):  {dit['trunk']['total_bytes']:,} bytes ({dit['trunk']['total_bytes']/1e9:.3f} GB)")
    print(f"  AdaLN Total (50 blocks):      {dit['adaln']['total_bytes']:,} bytes ({dit['adaln']['total_bytes']/1e9:.3f} GB)")
    print(f"    - AdaLN linear weights:     {dit['adaln']['weight_bytes']:,} bytes ({dit['adaln']['weight_bytes']/1e9:.3f} GB)")
    print(f"    - AdaLN linear biases:      {dit['adaln']['bias_bytes']:,} bytes ({dit['adaln']['bias_bytes']/1e9:.3f} GB)")
    print(f"  Run-Once Components:          {dit['run_once']['total_bytes']:,} bytes ({dit['run_once']['total_bytes']/1e9:.3f} GB)")

    print("\n[Per-Block Streaming Metrics]")
    print(f"  Total Per-Block Size:         {dit['per_block']['total_bytes']:,} bytes ({dit['per_block']['total_bytes']/(1024*1024):.2f} MiB)")
    print(f"  Trunk Per-Block (Max Run):    {dit['trunk']['max_block_run_bytes']:,} bytes ({dit['trunk']['max_block_run_bytes']/(1024*1024):.2f} MiB)")
    print(f"  2-Slot Ring Buffer Floor:     {dit['trunk']['ring_2slot_bytes']:,} bytes ({dit['trunk']['ring_2slot_bytes']/(1024*1024):.2f} MiB)")

    print("\n[DiT Canonical Groups]")
    print(f"  {'Group':<35} {'Bytes':>15} {'GB':>10} {'Count':>8} {'Types'}")
    print("  " + "-" * 74)
    for g_name, g in sorted(dit["groups"].items(), key=lambda x: -x[1]["bytes"]):
        types_str = ",".join(g["types"])
        print(f"  {g_name:<35} {g['bytes']:>15,} {g['bytes']/1e9:>9.3f}G {g['count']:>8} {types_str}")

    print("\n" + "=" * 78)
    print("Qwen3-VL Text Encoder (TE) Remote Census Summary")
    print("=" * 78)
    print(f"Total Tensors: {te['total_tensors']}")
    print(f"Total Size:    {te['total_bytes']:,} bytes ({te['total_bytes']/1e9:.3f} GB)")
    print(f"Architecture:  {te['arch']}")
    print(f"Layers < 50:   {te['layers_lt50']['bytes']:,} bytes ({te['layers_lt50']['bytes']/1e9:.3f} GB) [{te['layers_lt50']['layer_count']} layers, {te['layers_lt50']['tensor_count']} tensors]")
    print(f"Layers >= 50:  {te['layers_ge50']['bytes']:,} bytes ({te['layers_ge50']['bytes']/1e9:.3f} GB) [{te['layers_ge50']['tensor_count']} tensors]")
    print(f"Embed Tokens:  {te['embed_tokens']['bytes']:,} bytes ({te['embed_tokens']['bytes']/1e9:.3f} GB)")
    print(f"Visual Stack:  {te['visual']['bytes']:,} bytes ({te['visual']['bytes']/1e9:.3f} GB)")
    if te["other"]["bytes"]:
        print(f"Other:         {te['other']['bytes']:,} bytes ({te['other']['bytes']/1e9:.3f} GB)")

    print("\n" + "=" * 78)
    print("Reductions Ledger Table (DiT Bytes per Step)")
    print("=" * 78)
    print(f"| Stage                                                | DiT Bytes Touched per Step |")
    print(f"|---|---|")
    print(f"| Official BF16 Transformer                            | {ledger['official_bf16_gb']:.2f} GB |")
    print(f"| realrebelai Q4_K_M GGUF                              | {ledger['q4_k_m_gguf']['gb']:.2f} GB ({ledger['q4_k_m_gguf']['bytes']:,} B) |")
    print(f"| − AdaLN folded into exact table                      | {ledger['after_adaln_folded']['gb']:.2f} GB ({ledger['after_adaln_folded']['bytes']:,} B) |")
    print(f"| − Run-once components (refiner, condition proj)      | {ledger['trunk_per_step']['gb']:.2f} GB ({ledger['trunk_per_step']['bytes']:,} B) |")
    print(f"| RAM floor for DiT phase (2 ring slots + activations) | ≈ {ledger['ram_floor']['estimated_phase_gb']:.1f} GB (2 slots = {ledger['ram_floor']['ring_2slot_mb']:.2f} MB) |")
    print("=" * 78 + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description="Remote GGUF census for FastH3 and Qwen3-VL")
    parser.add_argument("--dit", default=DEFAULT_DIT_URL, help="URL or path to FastH3 DiT GGUF")
    parser.add_argument("--te", default=DEFAULT_TE_URL, help="URL or path to Qwen3-VL Text Encoder GGUF")
    parser.add_argument("--out", default=DEFAULT_OUTPUT_PATH, help="Output path for census.json")
    args = parser.parse_args()

    run_census(dit_url=args.dit, te_url=args.te, out_path=args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
