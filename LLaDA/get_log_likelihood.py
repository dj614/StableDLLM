import argparse

import torch
import torch.nn.functional as F

from model_utils import DEFAULT_LLADA_BASE, load_model_and_tokenizer, resolve_mask_id


def forward_process(batch, prompt_index, mask_id):
    b, l = batch.shape

    target_len = (l - prompt_index.sum()).item()
    k = torch.randint(1, target_len + 1, (), device=batch.device)

    x = torch.round(torch.linspace(float(k), k + (b - 1) * (target_len / b), steps=b, device=batch.device)).long()
    x = ((x - 1) % target_len) + 1
    assert x.min() >= 1 and x.max() <= target_len

    indices = torch.arange(target_len, device=batch.device).repeat(b, 1)
    is_mask = indices < x.unsqueeze(1)
    for i in range(b):
        is_mask[i] = is_mask[i][torch.randperm(target_len)]

    is_mask = torch.cat((torch.zeros(b, prompt_index.sum(), dtype=torch.bool, device=batch.device), is_mask), dim=1)
    noisy_batch = torch.where(is_mask, mask_id, batch)

    # Return the masked batch and the mask ratio
    return noisy_batch, (x / target_len).unsqueeze(1).repeat(1, l)


def get_logits(model, batch, prompt_index, cfg_scale, mask_id):
    if cfg_scale > 0.0:
        assert len(prompt_index) == batch.shape[1]
        prompt_index = prompt_index.unsqueeze(0).repeat(batch.shape[0], 1)
        un_batch = batch.clone()
        un_batch[prompt_index] = mask_id
        batch = torch.cat([batch, un_batch])

    logits = model(batch).logits

    if cfg_scale > 0.0:
        logits, un_logits = torch.chunk(logits, 2, dim=0)
        logits = un_logits + (cfg_scale + 1) * (logits - un_logits)
    return logits


@torch.no_grad()
def get_log_likelihood(model, prompt, answer, mc_num=128, batch_size=16, cfg_scale=0.0, mask_id=None):
    '''
    Monte Carlo likelihood estimate for diffusion-style masking models.

    Args:
        model: Mask predictor.
        prompt: (l1,) tensor of input ids.
        answer: (l2,) tensor of input ids.
        mc_num: Monte Carlo samples.
        batch_size: Mini-batch size.
        cfg_scale: Unsupervised classifier-free guidance scale.
        mask_id: The diffusion [MASK] token id. For MMaDA, resolve this from the tokenizer/config.
    '''
    if mask_id is None:
        mask_id = getattr(getattr(model, "config", None), "mask_token_id", None)
        if mask_id is None:
            raise ValueError("mask_id is required (or model.config.mask_token_id must exist)")
    mask_id = int(mask_id)

    device = next(model.parameters()).device

    seq = torch.concatenate([prompt, answer])[None, :]
    seq = seq.repeat((batch_size, 1)).to(device)
    prompt_index = torch.arange(seq.shape[1], device=device) < len(prompt)

    loss_ = []
    for _ in range(mc_num // batch_size):
        perturbed_seq, p_mask = forward_process(seq, prompt_index, mask_id)
        mask_index = perturbed_seq == mask_id

        logits = get_logits(model, perturbed_seq, prompt_index, cfg_scale, mask_id)

        loss = F.cross_entropy(logits[mask_index], seq[mask_index], reduction="none") / p_mask[mask_index]
        loss = loss.sum() / batch_size
        loss_.append(loss.item())

    return -sum(loss_) / len(loss_)


def main(argv=None, *, default_model_name_or_path: str = DEFAULT_LLADA_BASE):
    parser = argparse.ArgumentParser(description="Compute MC log-likelihood for LLaDA/MMaDA-style checkpoints")
    parser.add_argument("--model_name_or_path", type=str, default=default_model_name_or_path)
    parser.add_argument("--prompt", type=str, default="Roof shingle removal: A man is sitting on a roof. He")
    parser.add_argument("--answer", type=str, default=" is using wrap to wrap a pair of skis.")

    parser.add_argument("--mc_num", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--cfg_scale", type=float, default=0.0)

    parser.add_argument("--mask_id", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--dtype", type=str, default="bf16", choices=["bf16", "fp16", "fp32"])
    parser.add_argument("--no_trust_remote_code", action="store_true")

    args = parser.parse_args(argv)

    model, tokenizer, dev = load_model_and_tokenizer(
        args.model_name_or_path,
        device=args.device,
        dtype=args.dtype,
        trust_remote_code=not args.no_trust_remote_code,
    )

    mask_id = resolve_mask_id(tokenizer, model, override=args.mask_id)

    prompt_ids = torch.tensor(tokenizer(args.prompt)["input_ids"]).to(dev)
    answer_ids = torch.tensor(tokenizer(args.answer)["input_ids"]).to(dev)

    print(get_log_likelihood(model, prompt_ids, answer_ids, mc_num=args.mc_num, batch_size=args.batch_size, cfg_scale=args.cfg_scale, mask_id=mask_id))


if __name__ == "__main__":
    main()
