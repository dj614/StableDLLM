"""llada: lightweight inference & evaluation utilities.

This repo historically used a couple of different import layouts (e.g. `LLaDA.llada`).
To keep the codebase simple and predictable, `llada` is now a normal top-level
package under `src/`.

If you run scripts from repo root, make sure `src/` is on `PYTHONPATH`, e.g.:

  PYTHONPATH=src:$PYTHONPATH python -m llada.cli.main --help
"""

__all__ = ["cli", "eval", "model", "tasks", "utils"]
