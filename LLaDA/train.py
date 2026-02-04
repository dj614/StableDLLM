import os
import sys
from pathlib import Path


if "--china" in sys.argv:
    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
    os.environ["HF_HUB_ENDPOINT"] = "https://hf-mirror.com"
    os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"


# Allow running without installation by ensuring `src/` is on PYTHONPATH.
_REPO_ROOT = Path(__file__).resolve().parents[1]
_SRC = _REPO_ROOT / "src"
if _SRC.exists():
    sys.path.insert(0, str(_SRC))


from llada.plus.cli.train import parse_args  # noqa: E402
from llada.plus.train.runner import train  # noqa: E402


def main() -> None:
    args = parse_args()
    train(args)


if __name__ == "__main__":
    main()
