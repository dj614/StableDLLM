"""Output directory creation and broadcast for distributed runs."""

from __future__ import annotations

from pathlib import Path

from accelerate.utils import broadcast_object_list


def make_output_dir_and_broadcast(args, accelerator) -> Path:
    """Create the output dir and broadcast its path to all ranks.

    Keeps the same default naming scheme as the original script.
    """
    if accelerator.is_main_process and args.output_dir is None:
        args.output_dir = (
            f"/root/workspace/checkpoints/"
            f"seed{args.seed}_{args.model}_{args.task}_"
            f"ppots_mirror_plus"
        )
    args.output_dir = broadcast_object_list([args.output_dir])[0]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir
