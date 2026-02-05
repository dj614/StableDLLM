#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Merge predictions.jsonl with a ground-truth jsonl.

This is intentionally lightweight and filesystem-friendly:
- no hard-coded paths
- supports matching by an id key (default: "id"), falling back to meta.index then line index.

Run from repo root:
  python src/tools/eval/merge_predictions_with_gt.py --pred_jsonl ... --gt_jsonl ... --out_jsonl ...
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

# Allow running as a script from repo root without installing the package.
_REPO_ROOT = Path(__file__).resolve().parents[3]
_SRC_DIR = _REPO_ROOT / "src"
for _p in (str(_SRC_DIR), str(_REPO_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

<<<<<<< HEAD
from mdm.utils.io import iter_jsonl, write_jsonl  # noqa: E402
=======
from core.utils.io import iter_jsonl, write_jsonl  # noqa: E402
>>>>>>> 31bc6818f4abfc6e39eea2cd09727693801ec40c


def _get_id(row: Dict[str, Any], key: str) -> Optional[str]:
    if key in row:
        return str(row[key])
    meta = row.get("meta")
    if isinstance(meta, dict) and key in meta:
        return str(meta[key])
    if isinstance(meta, dict) and "index" in meta:
        return str(meta["index"])
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred_jsonl", type=str, required=True)
    ap.add_argument("--gt_jsonl", type=str, required=True)
    ap.add_argument("--out_jsonl", type=str, required=True)
    ap.add_argument("--id_key", type=str, default="id", help="match key; fallback to meta.index then line index")
    args = ap.parse_args()

    preds = list(iter_jsonl(args.pred_jsonl))
    gts = list(iter_jsonl(args.gt_jsonl))

    gt_map: Dict[str, Dict[str, Any]] = {}
    for i, g in enumerate(gts):
        gid = _get_id(g, args.id_key) or str(i)
        gt_map[gid] = g

    merged: List[Dict[str, Any]] = []
    missing = 0
    for i, p in enumerate(preds):
        pid = _get_id(p, args.id_key) or str(i)
        g = gt_map.get(pid)
        row = dict(p)
        if g is None:
            missing += 1
        else:
            if "gold_raw" not in row:
                row["gold_raw"] = (
                    g.get("gold_raw")
                    or g.get("answer")
                    or g.get("output")
                    or g.get("ground_truth")
                    or ""
                )
            if "prompt" not in row:
                row["prompt"] = g.get("prompt") or g.get("question") or g.get("input") or ""
            row["gt_meta"] = {k: v for k, v in g.items() if k not in ("gold_raw", "prompt")}
        merged.append(row)

    out = Path(args.out_jsonl)
    write_jsonl(out, merged)
    print(f"wrote: {out} (n={len(merged)}), missing_gt={missing}")


if __name__ == "__main__":
    main()
