"""Output directory helpers.

The legacy LLaDA training scripts created a default output directory on the
main process and broadcasted it to all ranks. The refactor moved that behavior
behind :func:`make_output_dir_and_broadcast`.
"""

from __future__ import annotations

from pathlib import Path

from accelerate.utils import broadcast_object_list


def _default_output_dir(args) -> str:
    """Construct a stable default output directory string."""
    seed = getattr(args, "seed", "na")
    model = getattr(args, "model", "model")
    task = getattr(args, "task", "task")

    suffix = []
    if bool(getattr(args, "PPOTS", False)):
        suffix.append("ppots")
    mode = str(getattr(args, "train_mode", ""))
    if mode and mode.lower() not in {"normal", "none"}:
        suffix.append(mode.lower())

    suf = ("_" + "_".join(suffix)) if suffix else ""
    return f"/root/workspace/checkpoints/seed{seed}_{model}_{task}{suf}"


def make_output_dir_and_broadcast(args, accelerator):
    """Create/broadcast ``args.output_dir`` and return it as a :class:`Path`."""

    if accelerator.is_main_process and getattr(args, "output_dir", None) is None:
        args.output_dir = _default_output_dir(args)

    # Make sure every rank sees the same output_dir.
    args.output_dir = broadcast_object_list([args.output_dir])[0]
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    return out
