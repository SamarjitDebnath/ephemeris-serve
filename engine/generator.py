import asyncio
from typing import Protocol, Sequence

import torch
from engine.model_loader import model_loader
from tokenizer.tokenizer_service import tokenizer_service
from utils.stop_sequences import find_stop_index
from settings.settings import model_settings, logging_settings
from logger import setup_logger

logger = setup_logger(__name__, level=logging_settings.log_level, log_file=logging_settings.log_file)


class GenerationRequest(Protocol):
    """Structural interface `InferenceEngine.generate_batch` depends on.

    Decouples the engine from the scheduler package -- any object with this
    shape (e.g. `scheduler.request.InferenceRequest`) can be batched, without
    the engine importing scheduler-owned types.
    """
    future: asyncio.Future
    queue: "asyncio.Queue | None"
    temperature: float
    top_k: int
    top_p: float
    max_tokens: int
    generated_tokens: list[int]
    finished: bool
    stop_sequences: list[str]


class InferenceEngine:
    def __init__(self):
        self.device = model_settings.device
        # Defer actual model retrieval/loading until first use to avoid
        # initializing torch/model state at import time (prevents semaphore
        # leaks and is safe with multi-worker servers).
        self._model = None

    @property
    def model(self):
        if self._model is None:
            self._model = model_loader._get_model()
        return self._model

    def invalidate_model_cache(self) -> None:
        """Drop the cached model reference so the next `.model` access
        re-fetches from `model_loader` -- used after `model_loader.reload()`
        swaps the underlying weights out from under an already-running
        server (see `scheduler/model_swap.py`)."""
        self._model = None

    def sample(self, logits, temperature, top_k=0, top_p=1.0):
        if temperature <= 0:
            return torch.argmax(logits, dim=-1).unsqueeze(-1)
        
        logits = logits / temperature

        # Top-K
        if top_k > 0:
            top_k = min(max(top_k, 1), logits.size(-1))
            indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
            logits[indices_to_remove] = float('-inf')

        # Top-p
        if top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)

            sorted_indices_to_remove = cumulative_probs > top_p
            # Shift the indices to the right to keep also the first token above the threshold
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0

            indices_to_remove = torch.zeros_like(logits, dtype=torch.bool)
            indices_to_remove.scatter_(1, sorted_indices, sorted_indices_to_remove)
            logits[indices_to_remove] = float('-inf')

        probabilities = torch.softmax(logits, dim=-1)
        next_token = torch.multinomial(probabilities, num_samples=1)
        return next_token

    # -----------------------------------------------------------------
    # Composable building blocks for the continuous scheduler
    # -----------------------------------------------------------------

    def forward_step(
        self,
        input_ids,
        attention_mask,
        past_key_values=None,
        position_ids=None,
        logit_gather_indices=None,
    ):
        """Run a single forward pass through the model.

        Args:
            input_ids: Token IDs tensor ``(batch, seq_len)``.
            attention_mask: Attention mask tensor ``(batch, seq_len)``.
            past_key_values: Optional KV cache from a previous step.
            position_ids: Optional explicit per-token position ids,
                ``(batch, seq_len)``. Needed when the batch mixes rows with
                different real past lengths and/or padding (prefill/decode
                mixing) -- the model's implicit default position handling
                assumes a uniform past length across the batch, which isn't
                true in that case. ``None`` (the default) preserves today's
                implicit-position behavior for pure-decode/pure-prefill
                batches, where every row's positions are already uniform.
            logit_gather_indices: Optional per-row column index, ``(batch,)``,
                selecting which position's logits are the "real" next-token
                prediction for that row. Needed when a mixed batch's
                new-tokens region is right-padded (see prefill/decode mixing),
                so the real prediction isn't always at column ``-1`` for
                every row. ``None`` (the default) preserves today's
                unconditional last-column slice.

        Returns:
            Tuple of ``(logits, new_past_key_values)`` where *logits* has
            shape ``(batch, vocab_size)`` (last-position only).
        """
        if self.model is None:
            raise RuntimeError("Model failed to load")

        logger.debug(
            "Forward step input shapes: input_ids=%s attention_mask=%s past_key_values=%s position_ids=%s",
            tuple(input_ids.shape),
            tuple(attention_mask.shape) if attention_mask is not None else None,
            type(past_key_values).__name__ if past_key_values is not None else None,
            tuple(position_ids.shape) if position_ids is not None else None,
        )

        logger.info("Model device: %s, Model dtype: %s", self.model.device, self.model.dtype)

        # Inference-only: without this, every forward pass builds a full
        # autograd graph (activations retained across all layers) that
        # nothing ever calls .backward() on or otherwise releases -- a
        # steady per-step memory leak that torch.cuda/mps.empty_cache()
        # cannot reclaim, since the memory is genuinely referenced (by the
        # graph), not just cached. generate()/generate_batch() already wrap
        # their forward calls in this; this is the continuous scheduler's
        # only forward-pass entry point and was missing it.
        with torch.no_grad():
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                position_ids=position_ids,
                use_cache=True,
            )
            logits = outputs.logits

            # # Squeeze out any unexpected leading dimensions beyond (batch, seq, vocab)
            # while logits.dim() > 3:
            #     logits = logits.squeeze(1)
            if logits.dim() == 3:
                if logit_gather_indices is not None:
                    batch_indices = torch.arange(logits.shape[0], device=logits.device)
                    logits = logits[batch_indices, logit_gather_indices, :]
                else:
                    logits = logits[:, -1, :]
            elif logits.dim() != 2:
                logger.debug(
                    "Unexpected model logits shape: %s, outputs.past_key_values=%s",
                    tuple(logits.shape),
                    type(outputs.past_key_values).__name__,
                )
                raise RuntimeError(f"Unexpected logits shape from model: {tuple(logits.shape)}")

            return logits.clone(), outputs.past_key_values

    def apply_repetition_penalty(self, logits, input_ids, penalty=None):
        """Apply vectorized repetition penalty across the batch.

        Args:
            logits: Logits tensor ``(batch, vocab_size)``.
            input_ids: Per-row token history used to determine which tokens
                have already appeared. Either a dense ``(batch, seq_len)``
                tensor (every row the same real length -- e.g.
                ``generate_batch``'s uniformly-growing batch), or a sequence
                of per-row 1D tensors of possibly *different* lengths (e.g.
                the continuous scheduler's per-request full history, which
                doesn't share a common length across a mixed prefill/decode
                batch). Row ``i``'s real length must be its own -- never pass
                a step's new-tokens-only slice here, or the penalty only
                ever "sees" the most recent token(s) instead of everything
                generated so far.
            penalty: Multiplicative penalty factor.  Defaults to the value
                from ``model_settings.repetition_penalty``.

        Returns:
            The modified logits tensor (same object, mutated in-place).
        """
        if penalty is None:
            penalty = model_settings.repetition_penalty
        if penalty == 1.0:
            return logits

        if logits.dim() == 3:
            logits = logits[:, -1, :]
        if logits.dim() != 2:
            raise RuntimeError(f"Unexpected logits shape for repetition penalty: {tuple(logits.shape)}")

        for i in range(len(input_ids)):
            unique_tokens = torch.unique(input_ids[i])
            if unique_tokens.numel() == 0:
                continue
            valid_tokens = unique_tokens[unique_tokens < logits.size(-1)].to(logits.device)
            if valid_tokens.numel() == 0:
                continue

            selected_logits = logits[i, valid_tokens]
            penalized_logits = torch.where(
                selected_logits < 0,
                selected_logits * penalty,
                selected_logits / penalty,
            )
            logits[i, valid_tokens] = penalized_logits

        return logits

    @property
    def eos_token_id(self):
        """Return the model's end-of-sequence token ID."""
        if self.model is None:
            raise RuntimeError("Model failed to load")
        return self.model.config.eos_token_id
    
    # This method is just for rapid prototyping and testing. It is not used in the continuous scheduler.
    # Reason: It is slower and has no eviction/insertion logic for the KV cache, which is essential for continuous generation.
    def generate(self, input_ids, max_tokens: int = -1, temperature: float = -1.0):
        input_ids = input_ids.to(self.device)

        if max_tokens == -1:
            max_tokens = model_settings.max_length
        if temperature == -1.0:
            temperature = model_settings.temperature

        top_k = model_settings.top_k
        top_p = model_settings.top_p
        repetition_penalty = model_settings.repetition_penalty

        with torch.no_grad():
            for _ in range(max_tokens):
                if self.model is None:
                    raise RuntimeError("Model failed to load")
                outputs = self.model(input_ids)
                logits = outputs.logits[:, -1, :].clone()

                # Apply repetition penalty
                if repetition_penalty != 1.0:
                    for i in range(input_ids.shape[0]):
                        for token_id in set(input_ids[i].tolist()):
                            if logits[i, token_id] < 0:
                                logits[i, token_id] *= repetition_penalty
                            else:
                                logits[i, token_id] /= repetition_penalty

                next_token = self.sample(logits, temperature, top_k, top_p)

                if self.model and next_token[0, 0] == self.model.config.eos_token_id:
                    break
                
                if next_token.dim() == 1:
                    next_token = next_token.unsqueeze(-1)
                    
                input_ids = torch.cat([input_ids, next_token], dim=-1)

                yield next_token.item()

    # It is recommended to use the generate_batch method for batched generation in production.
    async def generate_batch(self, input_ids, attention_mask, requests: Sequence[GenerationRequest]):
        batch_size = input_ids.shape[0]
        if batch_size == 0:
            return [""] * len(requests)

        # Move tensors to correct device
        input_ids = input_ids.to(self.device)
        attention_mask = attention_mask.to(self.device)

        logger.info(f"Starting batched generation for {batch_size} requests.")

        # Active requests as (original_index, request) pairs -- avoids a
        # wrapper type that would duplicate/shadow state already on `requests`.
        active_requests: list[tuple[int, GenerationRequest]] = list(enumerate(requests))

        # Pre-allocate output list for all original requests
        outputs = [None] * len(requests)

        with torch.no_grad():
            past_key_values = None
            next_tokens = None

            while len(active_requests) > 0:
                active_requests = [(idx, r) for idx, r in active_requests if not r.future.cancelled()]
                if len(active_requests) == 0:
                    break
                if self.model is None:
                    raise RuntimeError("Model failed to load")
                
                # 1. Forward pass
                if past_key_values is None:
                    outputs_model = self.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        use_cache=True
                    )
                else:
                    outputs_model = self.model(
                        input_ids=next_tokens,
                        attention_mask=attention_mask,
                        past_key_values=past_key_values,
                        use_cache=True
                    )
                
                past_key_values = outputs_model.past_key_values
                logits = outputs_model.logits[:, -1, :].clone()
                
                # 2. Vectorized Batch Repetition Penalty
                if model_settings.repetition_penalty != 1.0:
                    if logits.dim() == 3:
                        logits = logits[:, -1, :]
                    for i in range(len(active_requests)):
                        unique_tokens = torch.unique(input_ids[i])
                        valid_tokens = unique_tokens[unique_tokens < logits.size(-1)].to(logits.device)
                        if valid_tokens.numel() == 0:
                            continue
                        selected_logits = logits[i, valid_tokens]

                        penalized_logits = torch.where(
                            selected_logits < 0,
                            selected_logits * model_settings.repetition_penalty,
                            selected_logits / model_settings.repetition_penalty
                        )

                        logits[i, valid_tokens] = penalized_logits
                
                # 3. Sample next tokens
                next_tokens = torch.zeros((len(active_requests), 1), dtype=torch.long, device=self.device)

                for i, (_, r) in enumerate(active_requests):
                    token = self.sample(
                        logits[i].unsqueeze(0),
                        r.temperature,
                        r.top_k,
                        r.top_p
                    )
                    next_tokens[i] = token

                # 4. Update request states and check for finished conditions
                keep_indices = []
                for i, (original_idx, r) in enumerate(active_requests):
                    token_id = next_tokens[i].item()
                    r.generated_tokens.append(token_id)

                    if getattr(r, "queue", None) is not None:
                        await r.queue.put(token_id)

                    # A stop sequence can span multiple tokens, so it's checked
                    # against the full decoded text rather than the new token alone.
                    stop_text = None
                    stop_sequences = getattr(r, "stop_sequences", None)
                    if stop_sequences:
                        decoded = tokenizer_service.decode(r.generated_tokens)
                        stop_idx = find_stop_index(decoded, stop_sequences)
                        if stop_idx is not None:
                            stop_text = decoded[:stop_idx]

                    if (
                        stop_text is not None
                        or token_id == self.model.config.eos_token_id
                        or len(r.generated_tokens) >= r.max_tokens
                    ):
                        r.finished = True
                        outputs[original_idx] = (
                            stop_text if stop_text is not None else tokenizer_service.decode(r.generated_tokens)
                        )
                    else:
                        keep_indices.append(i)
                
                # 5. Append new tokens to input_ids and update attention_mask
                input_ids = torch.cat([input_ids, next_tokens], dim=1)
                attention_mask = torch.cat(
                    [
                        attention_mask,
                        torch.ones((len(active_requests), 1), dtype=torch.long, device=self.device)
                    ],
                    dim=1
                )
                
                # 6. Compact batch if any request finished
                if len(keep_indices) < len(active_requests):
                    logger.info(f"Compacting batch: {len(active_requests)} active requests -> {len(keep_indices)} remaining active requests.")
                    if len(keep_indices) == 0:
                        break
                    active_indices = torch.tensor(keep_indices, dtype=torch.long, device=self.device)

                    input_ids = input_ids[active_indices]
                    attention_mask = attention_mask[active_indices]
                    next_tokens = next_tokens[active_indices]

                    # Compact past_key_values cache
                    if past_key_values is not None:
                        past_key_values.batch_select_indices(active_indices)

                    active_requests = [active_requests[idx] for idx in keep_indices]

            for i, req in enumerate(requests):
                if outputs[i] is None:
                    outputs[i] = tokenizer_service.decode(req.generated_tokens)

        # Send completion signal to streaming queues when the request has a live queue
        for request in requests:
            if getattr(request, "queue", None) is not None:
                await request.queue.put("[DONE]")

        return outputs


engine = InferenceEngine()
