"""lm-eval-harness wrapper for LLaDA/MMaDA-style diffusion LMs.

This file is inspired by the evaluation code from https://github.com/ML-GSAI/SMDM.

Key design choices:
  1) **Sampling/generation** delegates to `LLaDA/generate.py::generate()` to ensure
     evaluation uses the exact same diffusion decoding implementation as the
     standalone generation scripts.
  2) **Token ids** (especially the diffusion [MASK] id) are resolved from the
     tokenizer/model config so the same code works for both LLaDA and MMaDA
     checkpoints (MMaDA uses a different tokenizer).
"""

import accelerate
import torch
import random
import numpy as np
from datasets import Dataset
from tqdm import tqdm

from lm_eval.__main__ import cli_evaluate
from lm_eval.api.instance import Instance
from lm_eval.api.model import LM
from lm_eval.api.registry import register_model

from transformers import AutoModel, AutoTokenizer

from generate import generate
from get_log_likelihood import get_log_likelihood as mc_get_log_likelihood
from model_utils import resolve_mask_id


def set_seed(seed: int):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


@register_model("llada_dist")
class LLaDAEvalHarness(LM):
    def __init__(
        self,
        model_path: str = "",
        mask_id: int | None = None,
        max_length: int = 4096,
        batch_size: int = 32,
        mc_num: int = 128,
        is_check_greedy: bool = True,
        cfg: float = 0.0,
        steps: int = 1024,
        gen_length: int = 1024,
        block_length: int = 1024,
        remasking: str = "low_confidence",
        temperature: float = 0.0,
        device: str = "cuda",
        use_chat_template: bool = False,
        **kwargs,
    ):
        """Create an lm-eval-harness model wrapper.

        Args:
            model_path: HF model name or local path.
            mask_id: Override diffusion [MASK] id (if None, resolved from tokenizer/config).
            max_length: Max sequence length.
            batch_size: Mini-batch size used for MC likelihood estimation.
            mc_num: Monte Carlo estimation iterations.
            is_check_greedy: Whether lm-eval-harness should ask for greedy checks.
            cfg: Unsupervised classifier-free guidance scale.
            steps/gen_length/block_length/remasking/temperature: generation hyperparameters.
            use_chat_template: If True, wrap each prompt as a single user turn via
                `tokenizer.apply_chat_template(..., add_generation_prompt=True)`.
        """
        super().__init__()

        accelerator = accelerate.Accelerator()
        self.accelerator = accelerator if accelerator.num_processes > 1 else None

        model_kwargs = {}
        if self.accelerator is not None:
            model_kwargs.update({"device_map": {"": f"{self.accelerator.device}"}})

        raw_model = AutoModel.from_pretrained(
            model_path,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            **model_kwargs,
        )
        raw_model.eval()
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

        # Resolve special token IDs before accelerator wrapping.
        self.mask_id = resolve_mask_id(self.tokenizer, raw_model, override=mask_id)

        self.device = torch.device(device)
        if self.accelerator is not None:
            self.model = self.accelerator.prepare(raw_model)
            self.device = torch.device(f"{self.accelerator.device}")
            self._rank = self.accelerator.local_process_index
            self._world_size = self.accelerator.num_processes
        else:
            self.model = raw_model.to(device)
            self._rank = 0
            self._world_size = 1

        self.mc_num = int(mc_num)
        self.batch_size = int(batch_size)
        assert self.mc_num % self.batch_size == 0
        self.max_length = int(max_length)
        self.is_check_greedy = bool(is_check_greedy)

        self.cfg = float(cfg)
        self.steps = int(steps)
        self.gen_length = int(gen_length)
        self.block_length = int(block_length)
        self.remasking = str(remasking)
        self.temperature = float(temperature)

        self.use_chat_template = bool(use_chat_template)

    @property
    def rank(self):
        return self._rank

    @property
    def world_size(self):
        return self._world_size

    @property
    def eot_token_id(self):
        return self.tokenizer.eos_token_id

    def _encode_pair(self, context: str, continuation: str):
        """Tokenize a (context, continuation) pair with whitespace-preserving behavior."""
        n_spaces = len(context) - len(context.rstrip())
        if n_spaces > 0:
            continuation = context[-n_spaces:] + continuation
            context = context[:-n_spaces]

        whole_enc = self.tokenizer(context + continuation)["input_ids"]
        context_enc = self.tokenizer(context)["input_ids"]

        context_enc_len = len(context_enc)
        continuation_enc = whole_enc[context_enc_len:]
        return context_enc, continuation_enc

    @torch.no_grad()
    def get_loglikelihood(self, prefix: torch.Tensor, target: torch.Tensor) -> float:
        return float(
            mc_get_log_likelihood(
                self.model,
                prefix.to(self.device),
                target.to(self.device),
                mc_num=self.mc_num,
                batch_size=self.batch_size,
                cfg_scale=self.cfg,
                mask_id=self.mask_id,
            )
        )

    @torch.no_grad()
    def suffix_greedy_prediction(self, prefix: torch.Tensor, target: torch.Tensor) -> bool:
        """Greedy verification using the **same** diffusion decoding as `generate()`.

        lm-eval-harness uses this for a few benchmarks (e.g. LAMBADA). If you don't
        need it, set `is_check_greedy=False` to speed things up.
        """
        if not self.is_check_greedy:
            return False

        prefix = prefix.to(self.device)
        target = target.to(self.device)
        gen_len = int(target.numel())
        if gen_len <= 0:
            return True

        # Make decoding deterministic.
        out = generate(
            self.model,
            prefix.unsqueeze(0),
            steps=gen_len,
            gen_length=gen_len,
            block_length=gen_len,
            temperature=0.0,
            cfg_scale=self.cfg,
            remasking="low_confidence",
            mask_id=self.mask_id,
        )
        pred = out[0, prefix.numel() :]
        return bool(torch.equal(pred, target))

    def loglikelihood(self, requests: list[Instance]):
        def _tokenize(e):
            prefix, target = self._encode_pair(e["prefix"], e["target"])
            return {
                "prefix_text": e["prefix"],
                "target_text": e["target"],
                "prefix": prefix,
                "target": target,
            }

        ds = [{"prefix": req.args[0], "target": req.args[1]} for req in requests]
        ds = Dataset.from_list(ds)
        ds = ds.map(_tokenize)
        ds = ds.with_format("torch")

        prompt_len = [len(x["prefix"]) + len(x["target"]) for x in ds]
        assert max(prompt_len) <= self.max_length

        out = []
        with torch.no_grad():
            for elem in tqdm(ds, desc="Computing likelihood..."):
                prefix = elem["prefix"]
                target = elem["target"]

                ll = self.get_loglikelihood(prefix, target)
                is_target_greedy_dec = self.suffix_greedy_prediction(prefix, target)
                out.append((ll, 1.0 if is_target_greedy_dec else 0.0))

        try:
            torch.cuda.empty_cache()
        except Exception:
            pass
        return out

    def loglikelihood_rolling(self, requests: list[Instance]):
        raise NotImplementedError

    def _maybe_apply_chat_template(self, prompt_text: str) -> str:
        if not self.use_chat_template:
            return prompt_text
        if hasattr(self.tokenizer, "apply_chat_template"):
            messages = [{"role": "user", "content": prompt_text}]
            return self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        return prompt_text

    def generate_until(self, requests: list[Instance]):
        def _tokenize(e):
            question = self._maybe_apply_chat_template(e["question"])
            return {
                "question": self.tokenizer(question)["input_ids"],
                "question_text": question,
                "until": e["until"],
            }

        ds = [{"question": req.args[0], "until": req.args[1]["until"]} for req in requests]
        ds = Dataset.from_list(ds)
        ds = ds.map(_tokenize)
        ds = ds.with_format("torch")

        out = []
        for elem in tqdm(ds, desc="Generating..."):
            prompt = elem["question"].unsqueeze(0).to(self.device)
            stop_tokens = elem["until"]

            generated_answer = generate(
                self.model,
                prompt,
                steps=self.steps,
                gen_length=self.gen_length,
                block_length=self.block_length,
                temperature=self.temperature,
                cfg_scale=self.cfg,
                remasking=self.remasking,
                mask_id=self.mask_id,
            )

            generated_answer = self.tokenizer.decode(generated_answer[0][prompt.shape[1] :], skip_special_tokens=False)
            for stop_seq in stop_tokens:
                if stop_seq in generated_answer:
                    generated_answer = generated_answer.split(stop_seq)[0]

            # remove special tokens
            generated_answer_ids = self.tokenizer(generated_answer)["input_ids"]
            generated_answer = self.tokenizer.decode(generated_answer_ids, skip_special_tokens=True)
            out.append(generated_answer)

            if self.accelerator is not None:
                self.accelerator.wait_for_everyone()

        return out


@register_model("mmada_dist")
class MMaDAEvalHarness(LLaDAEvalHarness):
    """Alias of `llada_dist` for convenience when evaluating MMaDA checkpoints."""

    pass


if __name__ == "__main__":
    set_seed(1234)
    cli_evaluate()
