"""
Direction-stability analysis for the refusal-direction pipeline.

Motivation
----------
Arditi et al. show refusal is mediated by a single direction in the residual stream.
This project asks whether that direction is *shared* across conditions (e.g. monolingual
English vs. code-switched inputs) or *condition-specific*. That question is answered by
comparing the refusal directions extracted under each condition.

This tool covers the STATIC geometry part (no GPU required): it operates purely on the
artifacts the pipeline already writes for each run:
  - direction.pt                     the selected refusal direction (shape [d_model])
  - direction_metadata.json          {"layer": L, "pos": P} that was selected
  - generate_directions/mean_diffs.pt  all candidate directions [n_pos, n_layers, d_model]

It reports:
  1. Cosine similarity between each condition's SELECTED direction.
  2. Cosine similarity at a FIXED (layer, pos) coordinate across conditions, so you compare
     the same coordinate rather than each condition's own best (needs mean_diffs.pt).

IMPORTANT: cosine similarity is only meaningful WITHIN a single model (directions from
different models live in different vector spaces / dimensions). Compare SEA-LION-English
vs SEA-LION-codeswitch, not SEA-LION vs Qwen.

The RUNTIME transfer test (does condition A's direction, ablated, still suppress refusal
on condition B? — the causal cross-condition test) needs the model on a GPU and is left as
a stub below, wired to the pipeline's existing hooks.

Usage
-----
  python -m pipeline.analysis.direction_stability RUN_DIR [RUN_DIR ...] \
      [--labels en,cs_malay,cs_mandarin] [--layer L --pos P]
"""
import argparse
import json
import os

import numpy as np
import torch


def _load_selected_direction(run_dir: str) -> torch.Tensor:
    path = os.path.join(run_dir, "direction.pt")
    d = torch.load(path, map_location="cpu")
    return d.float().flatten()


def _load_meta(run_dir: str) -> dict:
    path = os.path.join(run_dir, "direction_metadata.json")
    return json.load(open(path)) if os.path.exists(path) else {}


def _load_mean_diffs(run_dir: str):
    path = os.path.join(run_dir, "generate_directions", "mean_diffs.pt")
    return torch.load(path, map_location="cpu").float() if os.path.exists(path) else None


def _cos(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a / (a.norm() + 1e-8)
    b = b / (b.norm() + 1e-8)
    return float((a @ b).item())


def cosine_matrix(vectors) -> np.ndarray:
    n = len(vectors)
    m = np.eye(n)
    for i in range(n):
        for j in range(n):
            m[i, j] = _cos(vectors[i], vectors[j])
    return m


def _print_matrix(m: np.ndarray, labels, title: str) -> None:
    print(f"\n=== {title} ===")
    w = max(8, max(len(str(l)) for l in labels))
    print(" " * (w + 1) + " ".join(f"{str(l)[:8]:>8}" for l in labels))
    for i, l in enumerate(labels):
        row = " ".join(f"{m[i, j]:8.3f}" for j in range(len(labels)))
        print(f"{str(l):>{w}} {row}")


def _same_dim(vectors) -> bool:
    return len({tuple(v.shape) for v in vectors}) == 1


def cross_condition_ablation_transfer(*args, **kwargs):
    """STUB (needs GPU + model). Causal cross-condition test:

    For directions {r_A} extracted per condition A, and eval prompts for condition B:
      - build ablation hooks from r_A  (pipeline.utils.hook_utils.get_all_direction_ablation_hooks)
      - generate completions on condition B's harmful prompts under those hooks
      - score refusal/ASR with the project judge
    A high off-diagonal ASR (A's direction suppresses refusal on B) => shared mechanism.
    A low one => the code-switched refusal is mediated by a different direction (the finding).

    Wire this to construct_model_base + generate_and_save_completions_for_dataset once the
    per-condition runs exist.
    """
    raise NotImplementedError("Runtime transfer test requires the model on a GPU; see docstring.")


def main() -> None:
    ap = argparse.ArgumentParser(description="Compare refusal directions across conditions.")
    ap.add_argument("run_dirs", nargs="+", help="pipeline run directories (each = one condition)")
    ap.add_argument("--labels", default=None, help="comma-separated labels, one per run dir")
    ap.add_argument("--layer", type=int, default=None, help="fixed layer for coordinate comparison")
    ap.add_argument("--pos", type=int, default=None, help="fixed position for coordinate comparison")
    args = ap.parse_args()

    labels = args.labels.split(",") if args.labels else [os.path.basename(d.rstrip("/")) for d in args.run_dirs]
    if len(labels) != len(args.run_dirs):
        ap.error(f"got {len(labels)} labels for {len(args.run_dirs)} run dirs")

    metas = [_load_meta(d) for d in args.run_dirs]
    print("Per-condition selected (layer, pos):")
    for l, m in zip(labels, metas):
        print(f"  {l:24} layer={m.get('layer')}, pos={m.get('pos')}")

    # 1) selected-direction cosine matrix
    sel = [_load_selected_direction(d) for d in args.run_dirs]
    if _same_dim(sel):
        _print_matrix(cosine_matrix(sel), labels, "Cosine similarity between SELECTED directions")
    else:
        dims = {l: tuple(v.shape) for l, v in zip(labels, sel)}
        print(f"\n[SELECTED directions have different dims {dims} -> different models; "
              "cosine similarity is undefined across models. Group runs by model.]")

    # 2) fixed-coordinate cosine matrix (compare the SAME (layer,pos) across conditions)
    mds = [_load_mean_diffs(d) for d in args.run_dirs]
    if all(md is not None for md in mds) and _same_dim([md.reshape(-1) for md in mds]):
        layer = args.layer if args.layer is not None else metas[0].get("layer")
        pos = args.pos if args.pos is not None else metas[0].get("pos")
        if layer is None or pos is None:
            print("\n[no layer/pos to fix (missing metadata); pass --layer/--pos]")
        else:
            try:
                vecs = [md[pos, layer] for md in mds]
                _print_matrix(cosine_matrix(vecs), labels,
                              f"Cosine similarity at FIXED coordinate (layer={layer}, pos={pos})")
            except Exception as e:  # noqa: BLE001 - report and continue
                print(f"\n[fixed-coordinate comparison skipped: {e}]")
    else:
        print("\n[mean_diffs.pt missing or mismatched across runs -> skipping fixed-coordinate comparison]")


if __name__ == "__main__":
    main()
