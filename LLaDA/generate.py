import argparse

import torch
import numpy as np
import torch.nn.functional as F

from model_utils import (
    DEFAULT_LLADA_INSTRUCT,
    format_user_prompt,
    load_model_and_tokenizer,
    resolve_mask_id,
)


def add_gumbel_noise(logits, temperature):
    '''
    Gumbel-max sampling helper.

    Returns a perturbed score tensor suitable for `argmax` sampling:
        argmax(logits / temperature + gumbel)

    Notes:
      - temperature <= 0 disables sampling noise (deterministic argmax)
      - we use float64 to reduce numerical issues (as suggested in arXiv:2409.02908)
    '''
    if temperature <= 0:
        return logits
    logits = logits.to(torch.float64)

    # Sample standard Gumbel noise: g = -log(-log(u))
    u = torch.rand_like(logits, dtype=torch.float64).clamp_(1e-12, 1.0 - 1e-12)
    g = -torch.log(-torch.log(u))
    return logits / float(temperature) + g


def get_num_transfer_tokens(mask_index, steps):
    '''
    In the reverse process, the interval [0, 1] is uniformly discretized into `steps` intervals.
    Because LLaDA employs a linear noise schedule (Eq. (8)), the expected number of tokens
    transitioned at each step should be consistent.

    This function precomputes how many masked tokens are transitioned at each step, per sample.
    '''
    if steps <= 0:
        raise ValueError("steps must be positive")

    # mask_index: [B, L_block] (bool)
    mask_num = mask_index.sum(dim=1)  # [B]
    base = mask_num // steps
    remainder = mask_num % steps

    B = mask_num.size(0)
    out = base[:, None].expand(B, steps).clone()
    if (remainder > 0).any():
        t = torch.arange(steps, device=mask_index.device)[None, :]
        out += (t < remainder[:, None]).to(out.dtype)
    return out.to(torch.int64)


@torch.no_grad()
def generate(
    model,
    prompt,
    steps=128,
    gen_length=128,
    block_length=128,
    temperature=0.0,
    cfg_scale=0.0,
    remasking="low_confidence",
    mask_id=None,
    attention_mask=None,
    return_logprobs=False,
):
    '''
    Args:
        model: Mask predictor.
        prompt: A tensor of shape (B, L), where B is batch size.
        steps: Sampling steps, less than or equal to gen_length.
        gen_length: Generated answer length.
        block_length: Block length, less than or equal to gen_length. If less than gen_length, it means using
            semi-autoregressive remasking.
        temperature: Categorical distribution sampling temperature.
        cfg_scale: Unsupervised classifier-free guidance scale.
        remasking: Remasking strategy. 'low_confidence' or 'random'.
        mask_id: The token id for the diffusion [MASK] token. Do NOT hard-code this for MMaDA; resolve from
            tokenizer/config instead.
        attention_mask: Optional attention mask for padded prompts.
        return_logprobs: Whether to return per-step log-probs (for analysis/debug).
    '''
    if mask_id is None:
        mask_id = getattr(getattr(model, "config", None), "mask_token_id", None)
        if mask_id is None:
            raise ValueError(
                "mask_id is required (or model.config.mask_token_id must exist). "
                "Use resolve_mask_id(tokenizer, model) and pass it in."
            )
    mask_id = int(mask_id)

    if attention_mask is not None and (attention_mask == 0).any():
        am = attention_mask.to(torch.bool)
        attention_bias = (am[:, :, None] & am[:, None, :]).unsqueeze(1)
    else:
        attention_bias = None

    batch_size = prompt.shape[0]
    x = torch.full((batch_size, prompt.shape[1] + gen_length), mask_id, dtype=torch.long).to(prompt.device)
    x[:, : prompt.shape[1]] = prompt.clone()

    prompt_index = (x != mask_id)

    assert gen_length % block_length == 0
    num_blocks = gen_length // block_length

    assert steps % num_blocks == 0
    steps = steps // num_blocks

    old_logps = [] if return_logprobs else None

    for num_block in range(num_blocks):
        block_mask_index = (
            x[
                :,
                prompt.shape[1] + num_block * block_length : prompt.shape[1] + (num_block + 1) * block_length,
            ]
            == mask_id
        )
        num_transfer_tokens = get_num_transfer_tokens(block_mask_index, steps)

        for i in range(steps):
            mask_index = (x == mask_id)

            if cfg_scale > 0.0:
                un_x = x.clone()
                un_x[prompt_index] = mask_id
                x_ = torch.cat([x, un_x], dim=0)

                # If we have padding-aware attention bias, replicate it for (x, un_x).
                attention_bias_ = None
                if attention_bias is not None:
                    attention_bias_ = attention_bias.repeat(2, 1, 1, 1)

                try:
                    logits = model(x_, attention_bias=attention_bias_).logits
                except TypeError:
                    logits = model(x_).logits

                logits, un_logits = torch.chunk(logits, 2, dim=0)
                logits = un_logits + (cfg_scale + 1) * (logits - un_logits)
            else:
                try:
                    logits = model(x, attention_bias=attention_bias).logits
                except TypeError:
                    logits = model(x).logits

            logits = logits.to(torch.float64)
            if 0 <= mask_id < logits.shape[-1]:
                logits[..., mask_id] = -float("inf")

            logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
            x0 = torch.argmax(logits_with_noise, dim=-1)  # [B, L]

            if remasking == "low_confidence":
                p = F.softmax(logits, dim=-1)
                x0_p = torch.squeeze(torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1)
            elif remasking == "random":
                x0_p = torch.rand((x0.shape[0], x0.shape[1]), device=x0.device)
            else:
                raise NotImplementedError(remasking)

            # Only allow sampling inside the current block.
            x0_p[:, prompt.shape[1] + (num_block + 1) * block_length :] = -np.inf

            x0 = torch.where(mask_index, x0, x)
            confidence = torch.where(mask_index, x0_p, -np.inf)
            transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)

            for j in range(confidence.shape[0]):
                _, select_index = torch.topk(confidence[j], k=num_transfer_tokens[j, i])
                transfer_index[j, select_index] = True

            if return_logprobs:
                lp = F.log_softmax(logits, dim=-1)
                sel = lp.gather(2, x0.unsqueeze(-1)).squeeze(-1)
                per_step_lp = (sel * mask_index).sum(dim=1) / mask_index.sum(dim=1).clamp(min=1)
                old_logps.append(per_step_lp)

            x[transfer_index] = x0[transfer_index]

    if return_logprobs:
        return x, old_logps
    return x


def main(argv=None, *, default_model_name_or_path: str = DEFAULT_LLADA_INSTRUCT):
    parser = argparse.ArgumentParser(description="Diffusion decoding for LLaDA/MMaDA-style checkpoints")
    parser.add_argument("--model_name_or_path", type=str, default=default_model_name_or_path)
    parser.add_argument(
        "--prompt",
        type=str,
        default=(
            "Lily can run 12 kilometers per hour for 4 hours. After that, she runs 6 kilometers per hour. "
            "How many kilometers can she run in 8 hours?"
        ),
    )

    parser.add_argument("--steps", type=int, default=128)
    parser.add_argument("--gen_length", type=int, default=128)
    parser.add_argument("--block_length", type=int, default=32)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--cfg_scale", type=float, default=0.0)
    parser.add_argument("--remasking", type=str, default="low_confidence", choices=["low_confidence", "random"])

    parser.add_argument("--mask_id", type=int, default=None, help="Override [MASK] token id")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--dtype", type=str, default="bf16", choices=["bf16", "fp16", "fp32"])
    parser.add_argument("--no_trust_remote_code", action="store_true")

    parser.add_argument(
        "--no_chat_template",
        action="store_true",
        help="Do not apply chat template; treat --prompt as raw text",
    )

    args = parser.parse_args(argv)

    model, tokenizer, dev = load_model_and_tokenizer(
        args.model_name_or_path,
        device=args.device,
        dtype=args.dtype,
        trust_remote_code=not args.no_trust_remote_code,
    )

    mask_id = resolve_mask_id(tokenizer, model, override=args.mask_id)

    prompt_text = args.prompt
    if not args.no_chat_template:
        prompt_text = format_user_prompt(tokenizer, prompt_text)

    inputs = tokenizer(prompt_text, return_tensors="pt")
    input_ids = inputs["input_ids"].to(dev)
    attention_mask = inputs.get("attention_mask")
    if attention_mask is not None:
        attention_mask = attention_mask.to(dev)

    out = generate(
        model,
        input_ids,
        steps=args.steps,
        gen_length=args.gen_length,
        block_length=args.block_length,
        temperature=args.temperature,
        cfg_scale=args.cfg_scale,
        remasking=args.remasking,
        mask_id=mask_id,
        attention_mask=attention_mask,
        return_logprobs=False,
    )

    print(tokenizer.batch_decode(out[:, input_ids.shape[1] :], skip_special_tokens=True)[0])


if __name__ == "__main__":
    main()
