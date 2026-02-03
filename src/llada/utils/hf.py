import os


def enable_hf_mirror_china() -> None:
    """Enable HuggingFace mirror endpoints commonly used in China.

    This keeps behavior compatible with existing scripts that set these env vars.
    """
    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
    os.environ.setdefault("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
