import argparse

import torch
import numpy as np
import torch.nn.functional as F

from model_utils import DEFAULT_LLADA_INSTRUCT, format_user_prompt, load_model_and_tokenizer, resolve_mask_id


def add_gumbel_noise(logits, temperature):
    '''
    Gumbel-max sampling helper.

    Returns a perturbed score tensor suitable for `argmax` sampling:
        argmax(logits / temperature + gumbel)

    Notes:
      - temperature <= 0 disables sampling noise (deterministic argmax).
      - we use float64 to reduce numerical issues (as suggested in arXiv:2409.02908).
    '''
    if temperature <= 0:
        return logits

    logits = logits.to(torch.float64)
    u = torch.rand_like(logits, dtype=torch.float64).clamp_(1e-12, 1.0 - 1e-12)
    g = -torch.log(-torch.log(u))
    return logits / float(temperature) + g


def get_num_transfer_tokens(mask_index, steps):
    '''
    Evenly distribute the number of tokens to transition across `steps`.

    Returns:
      (B, steps) int64 tensor indicating how many tokens to reveal each step.
    '''
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
    tokenizer,
    *,
    steps: int = 128,
    gen_length: int = 128,
    block_length: int = 128,
    temperature: float = 0.0,
    cfg_scale: float = 0.0,
    remasking: str = 'low_confidence',
    mask_id: int | None = None,
    trace_path: str | None = None,
):
    '''
    A version of sampling that logs intermediate generations to `trace_path`.

    This is primarily for visualization/debugging.
    '''
    if mask_id is None:
        mask_id = getattr(getattr(model, "config", None), "mask_token_id", None)
        if mask_id is None:
            raise ValueError("mask_id is required (or model.config.mask_token_id must exist)")
    mask_id = int(mask_id)

    device = next(model.parameters()).device

    x = torch.full((1, prompt.shape[1] + gen_length), mask_id, dtype=torch.long, device=device)
    x[:, :prompt.shape[1]] = prompt.clone().to(device)

    prompt_index = (x != mask_id)

    # Ensure block_length is valid
    block_length = int(block_length)
    if block_length <= 0:
        block_length = gen_length
    if block_length > gen_length:
        block_length = gen_length

    # Calculate number of blocks
    num_blocks = gen_length // block_length
    if gen_length % block_length != 0:
        num_blocks += 1

    # Adjust steps per block
    steps_per_block = max(1, steps // num_blocks)

    trace_f = open(trace_path, "a", encoding="utf-8") if trace_path else None

    print_i = 0
    for num_block in range(num_blocks):
        block_start = prompt.shape[1] + num_block * block_length
        block_end = min(prompt.shape[1] + (num_block + 1) * block_length, prompt.shape[1] + gen_length)

        block_mask_index = (x[:, block_start:block_end] == mask_id)
        num_transfer_tokens = get_num_transfer_tokens(block_mask_index, steps_per_block)

        for i in range(steps_per_block):
            mask_index = (x == mask_id)

            if cfg_scale > 0.:
                un_x = x.clone()
                un_x[prompt_index] = mask_id
                x_ = torch.cat([x, un_x], dim=0)
                try:
                    logits = model(x_).logits
                except TypeError:
                    logits = model(x_).logits
                logits, un_logits = torch.chunk(logits, 2, dim=0)
                logits = un_logits + (cfg_scale + 1) * (logits - un_logits)
            else:
                logits = model(x).logits

            logits = logits.to(torch.float64)
            if 0 <= mask_id < logits.shape[-1]:
                logits[..., mask_id] = -float("inf")

            logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
            x0 = torch.argmax(logits_with_noise, dim=-1)

            if remasking == 'low_confidence':
                p = F.softmax(logits, dim=-1)
                x0_p = torch.squeeze(torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1)
            elif remasking == 'random':
                x0_p = torch.rand((x0.shape[0], x0.shape[1]), device=x0.device)
            else:
                raise NotImplementedError(remasking)

            # Don't sample beyond the current block end.
            x0_p[:, block_end:] = -np.inf

            x0 = torch.where(mask_index, x0, x)
            confidence = torch.where(mask_index, x0_p, -np.inf)

            transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)
            for j in range(confidence.shape[0]):
                _, select_index = torch.topk(confidence[j], k=int(num_transfer_tokens[j, i]))
                transfer_index[j, select_index] = True
            x[transfer_index] = x0[transfer_index]

            print_i += 1
            if trace_f is not None:
                generated_token_ids = x[0, prompt.shape[1]:]
                formatted_output = []
                for token_id in generated_token_ids:
                    decoded_token = tokenizer.decode(int(token_id)).replace("\n", " ")
                    decoded_token = decoded_token.replace("<|eot_id|>", " ").replace("<|endoftext|>", " ")
                    formatted_output.append(f"*{decoded_token}&")
                final_output = "".join(formatted_output).strip()
                print(f"{print_i}, {final_output}", file=trace_f, flush=True)

    if trace_f is not None:
        trace_f.close()

    return x


def main(argv=None, *, default_model_name_or_path: str = DEFAULT_LLADA_INSTRUCT):
    parser = argparse.ArgumentParser(description="Visualize sampling steps for LLaDA/MMaDA")
    parser.add_argument("--model_name_or_path", type=str, default=default_model_name_or_path)
    parser.add_argument("--prompt", type=str, default="Explain what artificial intelligence is.")
    parser.add_argument("--steps", type=int, default=64)
    parser.add_argument("--gen_length", type=int, default=64)
    parser.add_argument("--block_length", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--cfg_scale", type=float, default=0.0)
    parser.add_argument("--remasking", type=str, default="random", choices=["low_confidence", "random"])
    parser.add_argument("--trace_path", type=str, default="sample_process.txt")

    parser.add_argument("--mask_id", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--dtype", type=str, default="bf16", choices=["bf16", "fp16", "fp32"])
    parser.add_argument("--no_trust_remote_code", action="store_true")
    parser.add_argument("--no_chat_template", action="store_true")

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

    input_ids = tokenizer(prompt_text)['input_ids']
    input_ids = torch.tensor(input_ids).to(dev).unsqueeze(0)

    _ = generate(
        model,
        input_ids,
        tokenizer,
        steps=args.steps,
        gen_length=args.gen_length,
        block_length=args.block_length,
        temperature=args.temperature,
        cfg_scale=args.cfg_scale,
        remasking=args.remasking,
        mask_id=mask_id,
        trace_path=args.trace_path,
    )


if __name__ == '__main__':
    main()
