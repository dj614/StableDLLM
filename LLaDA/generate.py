import torch
import numpy as np
import torch.nn.functional as F

from transformers import AutoTokenizer, AutoModel

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


@ torch.no_grad()
def generate(model, prompt, steps=128, gen_length=128, block_length=128, temperature=0.,
             cfg_scale=0., remasking='low_confidence', mask_id=126336, attention_mask=None, return_logprobs=False):
    '''
    Args:
        model: Mask predictor.
        prompt: A tensor of shape (B, L), where B is batch size.
        steps: Sampling steps, less than or equal to gen_length.
        gen_length: Generated answer length.
        block_length: Block length, less than or equal to gen_length. If less than gen_length, it means using semi_autoregressive remasking.
        temperature: Categorical distribution sampling temperature.
        cfg_scale: Unsupervised classifier-free guidance scale.
        remasking: Remasking strategy. 'low_confidence' or 'random'.
        mask_id: The toke id of [MASK] is 126336.
    '''
    if attention_mask is not None and (attention_mask == 0).any():
        am = attention_mask.to(torch.bool)
        attention_bias = (am[:, :, None] & am[:, None, :]).unsqueeze(1)
    else:
        attention_bias = None
    batch_size = prompt.shape[0]
    x = torch.full((batch_size, prompt.shape[1] + gen_length), mask_id, dtype=torch.long).to(prompt.device)
    x[:, :prompt.shape[1]] = prompt.clone()

    prompt_index = (x != mask_id)

    assert gen_length % block_length == 0
    num_blocks = gen_length // block_length

    assert steps % num_blocks == 0
    steps = steps // num_blocks

    old_logps = [] if return_logprobs else None
    for num_block in range(num_blocks):
        block_mask_index = (x[:, prompt.shape[1] + num_block * block_length: prompt.shape[1] + (num_block + 1) * block_length:] == mask_id)
        num_transfer_tokens = get_num_transfer_tokens(block_mask_index, steps)
        for i in range(steps):
            mask_index = (x == mask_id)
            if cfg_scale > 0.:
                un_x = x.clone()
                un_x[prompt_index] = mask_id
                x_ = torch.cat([x, un_x], dim=0)
                logits = model(x_).logits
                logits, un_logits = torch.chunk(logits, 2, dim=0)
                logits = un_logits + (cfg_scale + 1) * (logits - un_logits)
            else:
                logits = model(x, attention_bias=attention_bias).logits

            logits = logits.to(torch.float64)
            if 0 <= mask_id < logits.shape[-1]:
                logits[..., mask_id] = -float("inf")
            logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
            x0 = torch.argmax(logits_with_noise, dim=-1) # b, l

            if remasking == 'low_confidence':
                p = F.softmax(logits, dim=-1)
                x0_p = torch.squeeze(
                    torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1) # b, l
            elif remasking == 'random':
                x0_p = torch.rand((x0.shape[0], x0.shape[1]), device=x0.device)
            else:
                raise NotImplementedError(remasking)

            x0_p[:, prompt.shape[1] + (num_block + 1) * block_length:] = -np.inf

            x0 = torch.where(mask_index, x0, x)
            confidence = torch.where(mask_index, x0_p, -np.inf)
            transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)
            for j in range(confidence.shape[0]):
                _, select_index = torch.topk(confidence[j], k=num_transfer_tokens[j, i])
                transfer_index[j, select_index] = True

            if return_logprobs:
                # compute per‐step log‐prob for masked tokens under `model` (old policy)
                lp = F.log_softmax(logits, dim=-1)
                # gather log‐prob of selected tokens
                sel = lp.gather(2, x0.unsqueeze(-1)).squeeze(-1)
                # average over masked positions
                per_step_lp = (sel * mask_index).sum(dim=1) / mask_index.sum(dim=1).clamp(min=1)
                old_logps.append(per_step_lp)
            x[transfer_index] = x0[transfer_index]

    if return_logprobs:
        return x, old_logps
    return x


def main():
    device = 'cuda'

    model = AutoModel.from_pretrained('GSAI-ML/LLaDA-8B-Instruct', trust_remote_code=True, torch_dtype=torch.bfloat16).to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained('GSAI-ML/LLaDA-8B-Instruct', trust_remote_code=True)

    prompt = "Lily can run 12 kilometers per hour for 4 hours. After that, she runs 6 kilometers per hour. How many kilometers can she run in 8 hours?"

    # Add special tokens for the Instruct model. The Base model does not require the following two lines.
    m = [{"role": "user", "content": prompt}, ]
    prompt = tokenizer.apply_chat_template(m, add_generation_prompt=True, tokenize=False)

    input_ids = tokenizer(prompt)['input_ids']
    input_ids = torch.tensor(input_ids).to(device).unsqueeze(0)

    out = generate(model, input_ids, steps=128, gen_length=128, block_length=32, temperature=0., cfg_scale=0., remasking='low_confidence', return_logprobs=False)
    print(tokenizer.batch_decode(out[:, input_ids.shape[1]:], skip_special_tokens=True)[0])


if __name__ == '__main__':
    main()
