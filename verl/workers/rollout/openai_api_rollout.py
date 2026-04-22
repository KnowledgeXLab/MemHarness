# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
OpenAI-compatible HTTP chat completions for closed-model evaluation (scheme A).

Prompts are built with the reference tokenizer upstream; this rollout decodes them to
text, calls the API, and returns placeholder token tensors plus ``api_response_text``
in ``non_tensor_batch`` for env steps and logging.
"""

from __future__ import annotations

import json
import logging
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from tensordict import TensorDict

from verl import DataProto
from verl.utils.torch_functional import get_response_mask
from verl.workers.rollout.base import BaseRollout

logger = logging.getLogger(__name__)


def _normalize_chat_url(base_url: str) -> str:
    base = (base_url or "").rstrip("/")
    if not base.endswith("/v1"):
        base = base + "/v1"
    return base + "/chat/completions"


def _http_chat_completion(
    url: str,
    headers: Dict[str, str],
    payload: dict,
    timeout: float,
) -> str:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        err = e.read().decode("utf-8", errors="replace")[:800]
        raise RuntimeError(f"Chat API HTTP {e.code}: {err}") from e
    choices = body.get("choices") or []
    if not choices:
        return ""
    msg = choices[0].get("message") or {}
    content = msg.get("content")
    if content is None:
        return ""
    return content.strip()


class OpenAIApiRollout(BaseRollout):
    """Rollout via OpenAI-compatible ``/v1/chat/completions`` (no local generation)."""

    def __init__(self, config: DictConfig, tokenizer, **kwargs):
        super().__init__()
        self.config = config
        self.tokenizer = tokenizer
        self.oa = config.get("openai_api") or {}
        self._url = _normalize_chat_url(str(self.oa.get("base_url", "https://api.openai.com/v1")))
        self._api_key = str(self.oa.get("api_key", "") or "")
        self._model = str(self.oa.get("model", "gpt-4o-mini"))
        self._max_concurrent = int(self.oa.get("max_concurrent", 8))
        self._timeout = float(self.oa.get("timeout_sec", 120.0))
        self._max_retries = int(self.oa.get("max_retries", 2))
        self._retry_backoff = float(self.oa.get("retry_backoff_sec", 1.0))
        extra = self.oa.get("extra_headers")
        self._extra_headers: Dict[str, str] = dict(OmegaConf.to_container(extra, resolve=True)) if extra else {}
        body = self.oa.get("extra_body")
        self._extra_body: Dict[str, Any] = dict(OmegaConf.to_container(body, resolve=True)) if body else {}
        raw_disable_thinking = self.oa.get("disable_thinking", None)
        if raw_disable_thinking is not None:
            try:
                if bool(raw_disable_thinking):
                    self._extra_body.setdefault("enable_thinking", False)
            except Exception:
                pass
        else:
            # Qwen3 / Qwen3.5 compatible gateways often default to reasoning mode.
            # Unless the user explicitly overrides via extra_body, disable it here.
            model_norm = self._model.lower().replace("-", "").replace("_", "").replace(".", "")
            if "qwen3" in model_norm:
                self._extra_body.setdefault("enable_thinking", False)

    @torch.no_grad()
    def generate_sequences(self, prompts: DataProto, **kwargs) -> DataProto:
        idx = prompts.batch["input_ids"]
        attention_mask = prompts.batch["attention_mask"]
        position_ids = prompts.batch["position_ids"]
        eos_token_id = prompts.meta_info["eos_token_id"]
        pad_token_id = prompts.meta_info["pad_token_id"]

        batch_size = idx.size(0)
        device = idx.device

        if not self._api_key:
            raise ValueError(
                "openai_api.api_key is empty. Set actor_rollout_ref.rollout.openai_api.api_key "
                "or environment variable (e.g. OPENAI_API_KEY) via Hydra."
            )

        prompt_texts: List[str] = []
        for i in range(batch_size):
            row_mask = attention_mask[i].bool()
            ids = idx[i][row_mask].detach().cpu()
            prompt_texts.append(self.tokenizer.decode(ids, skip_special_tokens=False))

        do_sample = prompts.meta_info.get("do_sample", True)
        is_validate = prompts.meta_info.get("validate", False)
        if not do_sample:
            temperature = 0.0
            top_p = 1.0
        elif is_validate:
            temperature = float(self.config.val_kwargs.temperature)
            top_p = float(self.config.val_kwargs.top_p)
        else:
            temperature = float(self.config.temperature)
            top_p = float(self.config.top_p)

        if prompts.meta_info.get("temperature") is not None:
            temperature = float(prompts.meta_info["temperature"])
        if prompts.meta_info.get("top_p") is not None:
            top_p = float(prompts.meta_info["top_p"])

        max_tokens = int(prompts.meta_info.get("response_length", self.config.response_length))
        responses_text = self._complete_batch(prompt_texts, max_tokens=max_tokens, temperature=temperature, top_p=top_p)

        response = torch.full(
            (batch_size, max_tokens),
            pad_token_id,
            dtype=idx.dtype,
            device=device,
        )

        seq = torch.cat([idx, response], dim=-1)
        response_length = response.size(1)
        delta_position_id = torch.arange(1, response_length + 1, device=position_ids.device)
        delta_position_id = delta_position_id.unsqueeze(0).repeat(batch_size, 1)
        response_position_ids = position_ids[:, -1:] + delta_position_id
        position_ids = torch.cat([position_ids, response_position_ids], dim=-1)
        response_attention_mask = get_response_mask(response_id=response, eos_token=eos_token_id, dtype=attention_mask.dtype)
        attention_mask = torch.cat((attention_mask, response_attention_mask), dim=-1)

        batch = TensorDict(
            {
                "prompts": idx,
                "responses": response,
                "input_ids": seq,
                "attention_mask": attention_mask,
                "position_ids": position_ids,
            },
            batch_size=batch_size,
        )
        non_tensor = {"api_response_text": np.array(responses_text, dtype=object)}
        return DataProto(batch=batch, non_tensor_batch=non_tensor)

    def _complete_batch(
        self,
        prompt_texts: List[str],
        max_tokens: int,
        temperature: float,
        top_p: float,
    ) -> List[str]:
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self._api_key}",
            **self._extra_headers,
        }

        def one(i: int, user_text: str) -> str:
            payload: Dict[str, Any] = {
                "model": self._model,
                "messages": [{"role": "user", "content": user_text}],
                "max_tokens": max_tokens,
            }
            if self._extra_body:
                payload.update(self._extra_body)
            if temperature > 0:
                payload["temperature"] = temperature
                payload["top_p"] = top_p
            else:
                payload["temperature"] = 0

            delay = self._retry_backoff
            last_err: Optional[BaseException] = None
            for attempt in range(self._max_retries + 1):
                try:
                    return _http_chat_completion(self._url, headers, payload, self._timeout)
                except Exception as e:  # noqa: BLE001 — surface after retries
                    last_err = e
                    logger.warning("openai_api rollout attempt %s failed: %s", attempt, e)
                    if attempt < self._max_retries:
                        time.sleep(delay)
                        delay *= 2
            assert last_err is not None
            raise last_err

        if self._max_concurrent <= 1:
            return [one(i, t) for i, t in enumerate(prompt_texts)]

        out: List[Optional[str]] = [None] * len(prompt_texts)
        with ThreadPoolExecutor(max_workers=min(self._max_concurrent, len(prompt_texts))) as ex:
            futs = {ex.submit(one, i, prompt_texts[i]): i for i in range(len(prompt_texts))}
            for fut in as_completed(futs):
                i = futs[fut]
                out[i] = fut.result()
        return [x or "" for x in out]
