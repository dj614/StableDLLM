import argparse

import torch

from generate import generate
from model_utils import (
    DEFAULT_LLADA_INSTRUCT,
    format_user_prompt,
    load_model_and_tokenizer,
    resolve_mask_id,
    resolve_pad_id,
)


def chat(argv=None, *, default_model_name_or_path: str = DEFAULT_LLADA_INSTRUCT):
    parser = argparse.ArgumentParser(description="Interactive chat for LLaDA/MMaDA-style checkpoints")
    parser.add_argument("--model_name_or_path", type=str, default=default_model_name_or_path)
    parser.add_argument("--gen_length", type=int, default=128)
    parser.add_argument("--steps", type=int, default=128)
    parser.add_argument("--block_length", type=int, default=32)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--cfg_scale", type=float, default=0.0)
    parser.add_argument("--remasking", type=str, default="low_confidence", choices=["low_confidence", "random"])

    parser.add_argument("--mask_id", type=int, default=None, help="Override [MASK] token id")
    parser.add_argument("--pad_id", type=int, default=None, help="Override PAD token id (used for stripping padding)")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--dtype", type=str, default="bf16", choices=["bf16", "fp16", "fp32"])
    parser.add_argument("--no_trust_remote_code", action="store_true")

    parser.add_argument(
        "--no_chat_template",
        action="store_true",
        help="Do not apply chat template; treat each user message as raw text",
    )

    args = parser.parse_args(argv)

    model, tokenizer, dev = load_model_and_tokenizer(
        args.model_name_or_path,
        device=args.device,
        dtype=args.dtype,
        trust_remote_code=not args.no_trust_remote_code,
    )

    mask_id = resolve_mask_id(tokenizer, model, override=args.mask_id)
    pad_id = resolve_pad_id(tokenizer, model, override=args.pad_id)

    gen_length = args.gen_length
    steps = args.steps

    print("*" * 66)
    print(f"**  Answer Length: {gen_length}  |  Sampling Steps: {steps}  **")
    print("*" * 66)

    prompt = None
    conversation_num = 0

    while True:
        user_input = input("Enter your question: ").strip()
        if not user_input:
            continue

        user_text = user_input
        if not args.no_chat_template:
            user_text = format_user_prompt(tokenizer, user_text)

        input_ids = tokenizer(user_text, return_tensors="pt")["input_ids"].to(dev)

        if conversation_num == 0:
            prompt = input_ids
        else:
            # Typical chat templates start with a BOS token; skip it when appending.
            prompt = torch.cat([prompt, input_ids[:, 1:]], dim=1)

        out = generate(
            model,
            prompt,
            steps=steps,
            gen_length=gen_length,
            block_length=args.block_length,
            temperature=args.temperature,
            cfg_scale=args.cfg_scale,
            remasking=args.remasking,
            mask_id=mask_id,
        )

        answer = tokenizer.batch_decode(out[:, prompt.shape[1] :], skip_special_tokens=True)[0]
        print(f"Bot's reply: {answer}")

        # Strip padding tokens before the next turn.
        if pad_id is not None:
            prompt_1d = out[0]
            prompt = prompt_1d[prompt_1d != pad_id].unsqueeze(0)
        else:
            prompt = out

        conversation_num += 1
        print("-" * 71)


if __name__ == "__main__":
    chat()
