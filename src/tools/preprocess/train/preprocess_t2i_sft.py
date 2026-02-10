#!/usr/bin/env python
# coding: utf-8
"""text-to-image-2M ➜ processed SFT JSONL (prompt -> image-tokens).

Each output line:
  {"input_ids": [...], "prompt_length": int}

- prompt part: tokenized text (no special tokens by default) + optional user_suffix
- response part: <|soi|> + (VQ image tokens offset into extended vocab) + <|eoi|>

Streaming is used to avoid downloading the whole WebDataset locally.

Example:
  PYTHONPATH=src:. python src/tools/preprocess/train/preprocess_t2i_sft.py \
    --out_file ./data/train/text_to_image_2M_512_sft.jsonl \
    --tokenizer_path GSAI-ML/LLaDA-8B-Instruct \
    --vq_model_name showlab/magvitv2 \
    --max_samples 5000 \
    --max_length 4096 \
    --resolution 512
"""

import sys
from pathlib import Path

# Allow running from repo root without installing the package.
_REPO_ROOT = Path(__file__).resolve().parents[4]
_SRC_DIR = _REPO_ROOT / "src"
for _p in (str(_SRC_DIR), str(_REPO_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# Add MMaDA root so we can `from models import MAGVITv2`
_MMADA_DIR = _REPO_ROOT / "MMaDA"
if _MMADA_DIR.exists():
    if str(_MMADA_DIR) not in sys.path:
        sys.path.insert(0, str(_MMADA_DIR))

from mdm.utils.hf import maybe_enable_hf_mirror_china

import argparse
import json
import os

import torch
from datasets import load_dataset
from transformers import AutoTokenizer
from tqdm import tqdm

try:
    from PIL import Image  # noqa: F401
except Exception as e:
    raise RuntimeError("PIL is required. Please install pillow.") from e

try:
    from torchvision import transforms
except Exception as e:
    raise RuntimeError("torchvision is required for image transforms.") from e

# MMaDA VQ tokenizer (MAGVITv2)
try:
    from models import MAGVITv2  # type: ignore
except Exception as e:
    raise RuntimeError(
        "Failed to import MAGVITv2 from MMaDA. "
        "Make sure repo has MMaDA/ and you run with PYTHONPATH=src:."
    ) from e


# Reserved special token ids used in MMaDA (see MMaDA/training/prompting_utils.py)
SOI_ID = 126084  # <|soi|>
EOI_ID = 126085  # <|eoi|>


def _image_transform(resolution: int):
    # Match MMaDA/training/utils.py:image_transform (Resize -> CenterCrop -> ToTensor -> Normalize to [-1,1])
    return transforms.Compose(
        [
            transforms.Resize(resolution, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop((resolution, resolution)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True),
        ]
    )


def _pick_prompt(sample: dict) -> str:
    # text-to-image-2M sample usually has {"json": {"prompt": "..."}}
    js = sample.get("json", None)
    if isinstance(js, dict) and isinstance(js.get("prompt", None), str):
        return js["prompt"]
    if isinstance(sample.get("prompt", None), str):
        return sample["prompt"]
    # fallback: sometimes caption/text exists
    for k in ("text", "caption", "txt"):
        v = sample.get(k, None)
        if isinstance(v, str) and v.strip():
            return v
    return ""


def _pick_image(sample: dict):
    # WebDataset usually exposes one of: jpg/png/jpeg/webp
    for k in ("jpg", "png", "jpeg", "webp"):
        if k in sample:
            return sample[k]
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_file", type=str, default="./data/train/text_to_image_2M_512_sft.jsonl")
    ap.add_argument(
        "--tokenizer_path",
        type=str,
        default="GSAI-ML/LLaDA-8B-Instruct",
        help="Tokenizer name/path (HF repo id or local path). Must match the tokenizer used in training.",
    )
    ap.add_argument("--china", action="store_true", help="Use hf-mirror.com (mainland China mirror).")

    # Dataset streaming params (as in dataset card)
    ap.add_argument(
        "--base_url",
        type=str,
        default="https://huggingface.co/datasets/jackyhate/text-to-image-2M/resolve/main/data_512_2M/data_{i:06d}.tar",
        help="Shard URL template. Must contain '{i:06d}'.",
    )
    ap.add_argument("--num_shards", type=int, default=46, help="Number of tar shards.")
    ap.add_argument("--split", type=str, default="train", help="WebDataset split name (kept for symmetry).")

    # Output filtering
    ap.add_argument("--max_samples", type=int, default=5000, help="Write at most this many accepted samples.")
    ap.add_argument(
        "--max_length",
        type=int,
        default=4096,
        help="Keep only samples whose final input_ids length <= max_length.",
    )

    # Prompt formatting
    ap.add_argument(
        "--template",
        type=str,
        default="generate an image of the following description: {prompt}",
        help="Prompt template. Use '{prompt}' placeholder.",
    )
    ap.add_argument("--user_suffix", type=str, default="\n", help="Suffix appended after formatted prompt text.")

    # VQ tokenizer (image -> discrete tokens)
    ap.add_argument("--vq_model_name", type=str, default="showlab/magvitv2", help="HF repo id for MAGVITv2.")
    ap.add_argument("--resolution", type=int, default=512, help="Resize/Crop resolution fed into VQ encoder.")
    ap.add_argument("--batch_size", type=int, default=8, help="Batch size for VQ encoding.")
    ap.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Device for VQ encoding.",
    )
    args = ap.parse_args()

    if args.max_samples <= 0:
        raise ValueError("--max_samples must be positive")
    if args.max_length <= 0:
        raise ValueError("--max_length must be positive")

    maybe_enable_hf_mirror_china(args.china)
    os.makedirs(os.path.dirname(args.out_file) or ".", exist_ok=True)

    print(f"✓ Loading tokenizer: {args.tokenizer_path}")
    tok = AutoTokenizer.from_pretrained(args.tokenizer_path, use_fast=True, trust_remote_code=True)

    # Image token ids are offset into extended vocab after text vocab.
    image_token_offset = len(tok)
    print(f"✓ text vocab size = {image_token_offset} (image tokens will be shifted by this offset)")

    # Device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif args.device == "cuda":
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"✓ Using device: {device}")

    print(f"✓ Loading VQ model (MAGVITv2): {args.vq_model_name}")
    vq = MAGVITv2.from_pretrained(args.vq_model_name).to(device)
    vq.eval()
    for p in vq.parameters():
        p.requires_grad_(False)

    tfm = _image_transform(args.resolution)

    # Build streaming URLs
    urls = [args.base_url.format(i=i) for i in range(args.num_shards)]
    print(f"✓ Streaming WebDataset shards: {len(urls)} shards")

    dataset = load_dataset(
        "webdataset",
        data_files={"train": urls},
        split="train",
        streaming=True,
    )

    kept = 0
    seen = 0
    pbar = tqdm(total=args.max_samples, desc="Keeping (len<=max_length)")

    # small batching for VQ encoding
    batch_prompts = []
    batch_images = []

    def flush_batch(out_f):
        nonlocal kept, batch_prompts, batch_images

        if not batch_images:
            return

        # Encode prompts first (cheap) to avoid VQ work for obviously-too-long prompts.
        # We still need img token count; for MAGVITv2 at 512 and patch=16 it's typically 1024,
        # but to be safe we compute it after encoding.
        prompt_ids_list = []
        valid_mask = []  # which items are worth VQ encoding

        # (Conservative) assume image tokens <= 2048; prompt too huge will be skipped.
        # We will still enforce exact max_length after we get image token length.
        for pr in batch_prompts:
            text = args.template.format(prompt=pr)
            ids = tok.encode(text, add_special_tokens=False)
            if args.user_suffix:
                ids += tok.encode(args.user_suffix, add_special_tokens=False)
            prompt_ids_list.append(ids)
            # quick coarse filter: if prompt alone already exceeds max_length, skip
            valid_mask.append(len(ids) + 2 <= args.max_length)  # +2 for <|soi|>, <|eoi|> at least

        # If nothing valid, clear and return
        if not any(valid_mask):
            batch_prompts = []
            batch_images = []
            return

        # Build tensor batch only for valids
        imgs = []
        idx_map = []
        for i, ok in enumerate(valid_mask):
            if not ok:
                continue
            img = batch_images[i]
            if img is None:
                continue
            try:
                if hasattr(img, "convert"):
                    pil = img.convert("RGB")
                else:
                    # might be bytes
                    from io import BytesIO
                    from PIL import Image
                    pil = Image.open(BytesIO(img)).convert("RGB")
                imgs.append(tfm(pil))
                idx_map.append(i)
            except Exception:
                continue

        if not imgs:
            batch_prompts = []
            batch_images = []
            return

        pixel_values = torch.stack(imgs, dim=0).to(device)

        with torch.no_grad():
            codes = vq.get_code(pixel_values)  # (B, T) in [0, codebook_size)
        codes = codes.to(torch.long).cpu()
        # Offset into extended vocab
        codes = codes + int(image_token_offset)

        # Write out accepted examples
        for j in range(codes.shape[0]):
            if kept >= args.max_samples:
                break
            i = idx_map[j]
            prompt_ids = prompt_ids_list[i]
            img_ids = codes[j].tolist()

            input_ids = prompt_ids + [SOI_ID] + img_ids + [EOI_ID]
            # Exclude prompt + <|soi|> from loss/masking eligibility
            prompt_length = len(prompt_ids) + 1

            if len(input_ids) <= args.max_length:
                out_f.write(json.dumps({"input_ids": input_ids, "prompt_length": prompt_length}, ensure_ascii=False) + "\n")
                kept += 1
                pbar.update(1)

        batch_prompts = []
        batch_images = []

    with open(args.out_file, "w", encoding="utf8") as out_f:
        for sample in dataset:
            if kept >= args.max_samples:
                break
            seen += 1

            prompt = _pick_prompt(sample)
            if not prompt:
                continue
            img = _pick_image(sample)
            if img is None:
                continue

            batch_prompts.append(prompt)
            batch_images.append(img)

            if len(batch_images) >= args.batch_size:
                flush_batch(out_f)

        # flush tail
        if kept < args.max_samples:
            flush_batch(out_f)

    pbar.close()
    print(f"✓ Done: kept {kept} / seen {seen} streamed samples -> {args.out_file}")
    if kept < args.max_samples:
        print(f"⚠ Warning: only kept {kept} (< max_samples={args.max_samples}). "
              f"Try increasing max_length, or reduce filtering, or check dataset fields.")


if __name__ == "__main__":
    main()
