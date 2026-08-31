#!/usr/bin/env python3
"""Analyze an ik_llama.cpp expert-trace file (produced via --expert-trace).

Usage: python3 analyze_experts.py trace.bin [--top N] [--json out.json]

Prints per-layer expert usage counts (how often each expert was selected),
highlights the most common / rarest experts, and reports how much weight mass
the K most-used experts cover (useful for deciding which experts to keep in
RAM/VRAM vs stream from SSD).
"""
import argparse
import json
import struct
import sys
from collections import defaultdict


def read_trace(path):
    with open(path, "rb") as f:
        data = f.read()
    off = 0

    def take(fmt):
        nonlocal off
        size = struct.calcsize(fmt)
        vals = struct.unpack_from(fmt, data, off)
        off += size
        return vals

    magic = take("8s")[0]
    assert magic == b"IKEXP001", f"bad magic: {magic!r}"
    (n_layers,) = take("<I")
    layers = []
    for _ in range(n_layers):
        layer_idx, n_expert, n_topk = take("<III")
        layers.append({"layer": layer_idx, "n_expert": n_expert, "n_topk": n_topk})

    records = []
    while off < len(data):
        (n_tokens,) = take("<I")
        toks = [take("<ii") for _ in range(n_tokens)]  # (token_id, pos)
        per_token = []
        for _ in range(n_tokens):
            token_layers = []
            for li in layers:
                k = li["n_topk"]
                ids = take(f"<{k}i")
                wgts = take(f"<{k}f")
                token_layers.append((ids, wgts))
            per_token.append(token_layers)
        records.append({"tokens": toks, "layers": per_token})
    return layers, records


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("trace")
    ap.add_argument("--top", type=int, default=5, help="how many common/rare experts to show per layer")
    ap.add_argument("--json", help="dump full per-layer counts to this JSON file")
    ap.add_argument("--per-token", action="store_true", help="print expert ids for every token")
    args = ap.parse_args()

    layers, records = read_trace(args.trace)

    n_tokens = sum(len(r["tokens"]) for r in records)
    print(f"layers: {len(layers)}  tokens: {n_tokens}  records: {len(records)}")
    for li in layers:
        print(f"  layer {li['layer']}: n_expert={li['n_expert']} top_k={li['n_topk']}")

    if args.per_token:
        for r in records:
            for (tid, pos), token_layers in zip(r["tokens"], r["layers"]):
                sel = {l["layer"]: list(ids) for l, (ids, _) in zip(layers, token_layers)}
                print(f"pos={pos:6d} token={tid:8d} experts={sel}")
        return

    # per-layer usage counts and weight mass (tokens with ids[0] < 0 were never
    # routed through that layer, e.g. trimmed last-layer FFN during prefill)
    counts = [defaultdict(int) for _ in layers]
    wmass = [defaultdict(float) for _ in layers]
    not_routed = [0] * len(layers)
    for r in records:
        for token_layers in r["layers"]:
            for li, (ids, wgts) in zip(layers, token_layers):
                if ids[0] < 0:
                    not_routed[li["layer"]] += 1
                    continue
                for e, w in zip(ids, wgts):
                    counts[li["layer"]][e] += 1
                    wmass[li["layer"]][e] += w
    if any(not_routed):
        skipped = {layers[i]["layer"]: n for i, n in enumerate(not_routed) if n}
        print(f"note: tokens not routed through a layer (excluded): {skipped}")

    print(f"\nper-layer stats (over {n_tokens} tokens):")
    summary = {}
    for li, cnt, wm in zip(layers, counts, wmass):
        ne, k = li["n_expert"], li["n_topk"]
        by_use = sorted(cnt.items(), key=lambda kv: -kv[1])
        used = len(cnt)
        total_w = sum(wm.values()) or 1.0
        # coverage: fraction of routing weight carried by the K most-used experts
        cov = {}
        srt = sorted(wm.items(), key=lambda kv: -kv[1])
        for frac in (0.25, 0.5, 0.75):
            n_keep = max(1, int(ne * frac))
            cov[f"top{int(frac*100)}%"] = round(sum(w for _, w in srt[:n_keep]) / total_w, 4)
        summary[li["layer"]] = {
            "counts": {str(e): c for e, c in cnt.items()},
            "weight_mass": {str(e): round(w, 4) for e, w in wm.items()},
        }
        common = ", ".join(f"e{e}:{c}" for e, c in by_use[: args.top])
        rare = ", ".join(f"e{e}:{c}" for e, c in by_use[-args.top :][::-1])
        print(f"layer {li['layer']:3d}: {used}/{ne} experts ever used | coverage {cov}")
        print(f"    common: {common}")
        print(f"    rare  : {rare}")

    if args.json:
        with open(args.json, "w") as f:
            json.dump({"layers": layers, "n_tokens": n_tokens, "per_layer": summary}, f)
        print(f"\nfull counts written to {args.json}")


if __name__ == "__main__":
    main()
