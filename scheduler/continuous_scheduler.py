import asyncio
import time
import torch
from dataclasses import dataclass
from typing import List, Optional
from transformers import DynamicCache
from settings.settings import model_settings, logging_settings, cache_settings
from tokenizer.tokenizer_service import tokenizer_service
from scheduler.request_queue import request_queue
from scheduler.request import InferenceRequest
from utils.stop_sequences import find_stop_index
from utils.device_cache import empty_device_cache, maybe_empty_device_cache
from utils.errors import INTERNAL_ERROR_MESSAGE
from metrics.metrics import streaming_metrics
from cache.paged_kv_cache import PagedKVCache
from logger import setup_logger

logger = setup_logger(__name__, level=logging_settings.log_level, log_file=logging_settings.log_file)


@dataclass
class _BatchInputs:
    """One step's batched forward-pass inputs, built by `_prepare_batch`.

    A row's real "new tokens this step" occupy columns
    `[0, new_lengths[i])` of `input_ids`/the new-tokens region of
    `attention_mask` (right-padded, real content first) -- and, once
    concatenated after that row's `past_width`-wide past region, occupy
    `[past_width, past_width + new_lengths[i])` of the model's output
    `past_key_values`. See `ContinuousScheduler._dispatch_tokens`.
    """
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    position_ids: torch.Tensor
    logit_gather_indices: torch.Tensor
    past_key_values: Optional[DynamicCache]
    past_width: int
    new_lengths: List[int]


class ContinuousScheduler:
    """Continuously processes inference requests with dynamic batching.

    This scheduler maintains a pool of active requests and processes them token-by-token.
    New requests can be added at any time (subject to ``max_batch_size``). It leverages
    the model's KV cache (``past_key_values``) to avoid recomputing the full sequence for
    each step, enabling O(n) inference per request. Tokens are streamed back to the caller
    via an ``asyncio.Queue`` attached to each :class:`InferenceRequest`.
    """

    def __init__(self, engine, tokenizer, max_batch_size: int = 8, timeout: float = 0.01):
        self.engine = engine
        self.tokenizer = tokenizer
        self.max_batch_size = max_batch_size
        self.timeout = timeout
        # List of active ``InferenceRequest`` objects
        self.active_requests: list[InferenceRequest] = []
        # Lazily constructed once the model is loaded (see `paged_cache` property) --
        # building it here would force an eager model load.
        self._paged_cache: Optional[PagedKVCache] = None

    @property
    def paged_cache(self) -> PagedKVCache:
        """Block-based KV cache storage shared by every active request.

        Constructed on first access (not in __init__) so it doesn't force
        the model to load before it's otherwise needed.
        """
        if self._paged_cache is None:
            config = self.engine.model.config
            num_kv_heads = getattr(config, "num_key_value_heads", config.num_attention_heads)
            head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
            self._paged_cache = PagedKVCache(
                num_layers=config.num_hidden_layers,
                num_kv_heads=num_kv_heads,
                head_dim=head_dim,
                block_size=cache_settings.kv_block_size,
                dtype=self.engine.model.dtype,
                device=self.engine.device,
            )
        return self._paged_cache

    def invalidate_paged_cache(self) -> None:
        """Drop the paged KV cache so it rebuilds against the current
        model's architecture on next access.

        Used after a runtime model swap (see `scheduler/model_swap.py`):
        the old cache's tensor shapes (num_layers/num_kv_heads/head_dim/
        dtype) are tied to the previous model, and this must only be called
        once the caller has confirmed `active_requests` is empty -- an
        active request's `block_table` still points into the old cache.
        """
        self._paged_cache = None

    def _pad_batch(self, tensors, padding_value):
        """Right-pad each tensor to the batch's max width and stack them.

        Used for the "new tokens this step" region: real content starts at
        column 0 for every row, so once concatenated after that row's real
        past it stays one contiguous real range -- see `_prepare_batch`.
        """
        max_len = max(t.size(1) for t in tensors)
        padded = []
        for t in tensors:
            if t.dim() != 2:
                raise RuntimeError(f"Unexpected tensor shape in _pad_batch: {tuple(t.shape)}")
            if t.size(1) < max_len:
                pad_amt = max_len - t.size(1)
                padded.append(torch.nn.functional.pad(t, (0, pad_amt), value=padding_value))
            else:
                padded.append(t)
        return torch.cat(padded, dim=0)

    async def _add_new_requests(self):
        """Pull requests from the global ``request_queue`` until the batch is full
        or the timeout elapses.
        """
        while len(self.active_requests) < self.max_batch_size:
            try:
                req: InferenceRequest = await asyncio.wait_for(
                    request_queue.get(), timeout=self.timeout
                )
                # Tokenise the prompt once and move tensors to the engine device
                # Apply chat template if available
                formatted_prompt = req.prompt
                tokenizer_obj = self.tokenizer.tokenizer
                if hasattr(tokenizer_obj, 'apply_chat_template'):
                    try:
                        # Format as a single user message for chat models
                        messages = [{"role": "user", "content": req.prompt}]
                        formatted_prompt = tokenizer_obj.apply_chat_template(
                            messages,
                            tokenize=False,
                            add_generation_prompt=True
                        )
                        logger.info("Applied chat template to prompt: %s -> %s", req.prompt[:50], formatted_prompt[:80])
                    except Exception as e:
                        logger.warning("Failed to apply chat template: %s, using raw prompt", e)
                
                if req.deadline is not None and req.deadline <= time.monotonic():
                    logger.info("Dropping already-expired request before scheduling: prompt=%s", req.prompt)
                    streaming_metrics.record_timeout_eviction()
                    self._fail_request_timeout(req)
                    continue

                encoded = tokenizer_obj(
                    formatted_prompt,
                    return_tensors="pt",
                )
                req.input_ids = encoded["input_ids"].to(self.engine.device)
                # req.block_table starts empty (see InferenceRequest.__init__) --
                # that's the "no KV cache yet" signal for the first step.

                req.queue_latency_ms = time.monotonic() - req.enqueue_time
                streaming_metrics.record_queue_latency(req.queue_latency_ms)

                self.active_requests.append(req)
                logger.debug(
                    "Added request to scheduler: prompt=%s, active_requests=%d",
                    req.prompt,
                    len(self.active_requests),
                )
            except asyncio.TimeoutError:
                break

    def _prepare_batch(self) -> Optional[_BatchInputs]:
        """Build one batched forward-pass input, mixing prefill and decode rows.

        A request with no cached past yet contributes its whole prompt as
        new input; a request already mid-decode contributes exactly its one
        most-recently-generated token. Both kinds of rows are batched into
        the same step, so one request joining never forces every other
        active request to redo a full-sequence recompute.

        Returns ``None`` if there are no active requests to batch.
        """
        for req in self.active_requests:
            if req.block_table.length > 0 and not self.paged_cache.is_valid(req.block_table):
                logger.warning(
                    "Invalid paged KV state detected for prompt=%s; falling back to a fresh prefill.",
                    req.prompt,
                )
                self.paged_cache.free(req.block_table)

        if not self.active_requests:
            return None

        return self._build_batch_inputs(self.active_requests)

    def _build_batch_inputs(self, reqs: List[InferenceRequest]) -> Optional[_BatchInputs]:
        """Build one batched forward-pass input for an explicit request list.

        Factored out of `_prepare_batch` so a persistent whole-batch failure
        can retry a single request in its own batch of one -- see
        `_retry_requests_individually`. `_prepare_batch` itself still handles
        `self.active_requests` (with its cache-validity sweep); this just
        does the tensor construction, parameterized on whatever subset of
        requests is passed in.
        """
        if not reqs:
            return None

        new_token_tensors = [
            req.input_ids if req.block_table.length == 0 else req.input_ids[:, -1:]
            for req in reqs
        ]
        new_lengths = [t.shape[1] for t in new_token_tensors]
        max_new_len = max(new_lengths)

        keys_per_layer, values_per_layer, past_lengths = self.paged_cache.gather_dense(
            [req.block_table for req in reqs]
        )
        past_width = max(past_lengths) if past_lengths else 0

        # input_ids: new-tokens region, right-padded to max_new_len.
        input_ids = self._pad_batch(new_token_tensors, self.tokenizer.tokenizer.pad_token_id)

        # attention_mask: [past region, left-padded to past_width] +
        # [new-tokens region, right-padded to max_new_len].
        mask_rows = []
        for past_len, new_len in zip(past_lengths, new_lengths):
            past_part = torch.cat(
                [
                    torch.zeros((1, past_width - past_len), device=self.engine.device, dtype=torch.long),
                    torch.ones((1, past_len), device=self.engine.device, dtype=torch.long),
                ],
                dim=1,
            )
            new_part = torch.cat(
                [
                    torch.ones((1, new_len), device=self.engine.device, dtype=torch.long),
                    torch.zeros((1, max_new_len - new_len), device=self.engine.device, dtype=torch.long),
                ],
                dim=1,
            )
            mask_rows.append(torch.cat([past_part, new_part], dim=1))
        attention_mask = torch.cat(mask_rows, dim=0)

        # position_ids[i, j] = attention_mask[i, :past_width+j+1].sum() - 1,
        # clamped >= 0. Derived from the mask itself (not an assumed
        # layout), so it's correct regardless of which side padding sits on.
        cumulative = torch.cumsum(attention_mask, dim=1)
        position_ids = (cumulative[:, past_width:past_width + max_new_len] - 1).clamp(min=0)

        # logit_gather_indices[i]: column of the last real "new" token for row i,
        # *within the logits tensor* -- which spans only the new-tokens region
        # (width max_new_len), unlike past_key_values, which accumulates the
        # full past_width + max_new_len. Real content occupies [0, new_len)
        # since the new-tokens region is right-padded.
        logit_gather_indices = torch.tensor(
            [new_len - 1 for new_len in new_lengths],
            device=self.engine.device,
            dtype=torch.long,
        )

        if past_width == 0:
            past_key_values = None
        else:
            batched_past_layers = [
                (keys_per_layer[layer_idx], values_per_layer[layer_idx])
                for layer_idx in range(self.paged_cache.num_layers)
            ]
            past_key_values = DynamicCache(ddp_cache_data=batched_past_layers, config=self.engine.model.config)

        logger.debug(
            "Prepared mixed batch: past_width=%d max_new_len=%d past_lengths=%s new_lengths=%s",
            past_width,
            max_new_len,
            past_lengths,
            new_lengths,
        )

        return _BatchInputs(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            logit_gather_indices=logit_gather_indices,
            past_key_values=past_key_values,
            past_width=past_width,
            new_lengths=new_lengths,
        )

    async def _dispatch_tokens(self, reqs: List[InferenceRequest], next_tokens, new_past, past_width: int, new_lengths: List[int]):
        """Stream sampled tokens back to clients and update per-request state.

        Args:
            reqs: The requests this step's batch was built from, index-aligned
                with ``next_tokens``/``new_past``/``new_lengths`` -- normally
                ``self.active_requests``, but may be a single-request subset
                during isolation retries (see `_retry_requests_individually`).
            next_tokens: Tensor of shape ``(batch, 1)`` with one token per
                request in ``reqs``.
            new_past: The full batched ``past_key_values`` returned by the
                model's forward pass, width ``past_width + max(new_lengths)``.
            past_width: The shared past-region width fed into this step
                (``_BatchInputs.past_width``) -- the same for every row.
            new_lengths: Per-row count of real new tokens contributed this
                step (``_BatchInputs.new_lengths``) -- a row's whole prompt
                on its first step, or 1 while mid-decode.
        """
        logger.debug("Dispatching %d tokens", len(reqs))
        finished_requests = []
        for idx, req in enumerate(reqs):
            token_id = next_tokens[idx].item()

            # Stream token back to the client if the queue is configured
            if getattr(req, "queue", None) is not None:
                logger.debug("Sending token_id=%s to queue for prompt=%s", token_id, req.prompt)
                req.queue.put_nowait(token_id)

            # Record generated token
            req.generated_tokens.append(token_id)

            # Check whether the text generated so far contains one of the
            # request's stop sequences. Checked against the full decoded
            # text (not just this token) since a stop sequence can span
            # multiple tokens and needn't align with token boundaries.
            stop_text = None
            if req.stop_sequences:
                decoded = tokenizer_service.decode(req.generated_tokens)
                stop_idx = find_stop_index(decoded, req.stop_sequences)
                if stop_idx is not None:
                    stop_text = decoded[:stop_idx]

            # Append token to tensors for the next iteration
            # Ensure tensors are present and use matching dtype/device for concatenation
            assert req.input_ids is not None
            new_token = torch.tensor([[token_id]], dtype=req.input_ids.dtype, device=req.input_ids.device)
            req.input_ids = torch.cat([req.input_ids, new_token], dim=1)

            # Append this step's newly computed K/V into the paged store.
            # Real content for row `idx` sits at columns
            # [past_width : past_width + new_lengths[idx]] -- contiguous,
            # since `_prepare_batch` right-pads the new-tokens region (real
            # content first), immediately following that row's real past.
            new_kv = self._extract_new_kv(new_past, idx, past_width, new_lengths[idx], req.prompt)
            if new_kv is None:
                self.paged_cache.free(req.block_table)
            else:
                keys_per_layer, values_per_layer = new_kv
                self.paged_cache.append(req.block_table, keys_per_layer, values_per_layer)

            # Determine if the request has finished
            if (
                stop_text is not None
                or token_id == self.engine.eos_token_id
                or len(req.generated_tokens) >= req.max_tokens
            ):
                self._finish_request(req, final_text=stop_text)
                finished_requests.append(req)

        # Remove finished requests from the active pool
        self.active_requests = [r for r in self.active_requests if r not in finished_requests]

    def _extract_new_kv(self, new_past, idx: int, start_col: int, length: int, prompt: str):
        """Pull request `idx`'s newly computed (key, value) tensors out of `new_past`.

        `start_col`/`length` select the exact real-content range for this
        row (`[start_col : start_col + length]`) -- not just "the last N
        columns", since with a mixed batch the real new tokens for a
        shorter-than-max row sit before some trailing padding, not at the
        tensor's absolute end.

        Returns `(keys_per_layer, values_per_layer)` -- each a list of
        `(num_kv_heads, new_token_count, head_dim)` tensors, one per layer,
        ready for `PagedKVCache.append()` -- or `None` if `new_past` is
        missing or structurally invalid, in which case the caller should
        drop the request's cached state and let it recompute from scratch.
        """
        if new_past is None:
            logger.warning("No past_key_values returned for prompt=%s; dropping cached state.", prompt)
            return None

        keys_per_layer = []
        values_per_layer = []
        for layer_kv in new_past:
            # Allow an optional trailing None placeholder, like (key, value, None).
            if layer_kv is None or len(layer_kv) < 2 or any(kv is None for kv in layer_kv[:2]):
                logger.warning(
                    "Invalid past_key_values for prompt=%s; falling back to full prompt generation.",
                    prompt,
                )
                return None
            key_tensor, value_tensor = layer_kv[0], layer_kv[1]
            keys_per_layer.append(key_tensor[idx, :, start_col:start_col + length, :])
            values_per_layer.append(value_tensor[idx, :, start_col:start_col + length, :])

        return keys_per_layer, values_per_layer

    def _finish_request(self, req: InferenceRequest, final_text: str | None = None) -> None:
        """Resolve a request's future and signal end-of-stream to its queue.

        Shared by the natural EOS/max_tokens/stop-sequence finish path in
        `_dispatch_tokens` and by `_evict_expired_requests` (a timed-out
        request that already generated some tokens is finished with partial
        output, not failed).

        `final_text`, when given, overrides the decoded-tokens text used to
        resolve the request's future -- used when a stop sequence matched,
        so the sequence itself (and anything after it) is trimmed from the
        result instead of decoding the full, untrimmed token list.
        """
        req.finished = True
        logger.info(
            "Request finished for prompt=%s generated_length=%d finished=%s",
            req.prompt,
            len(req.generated_tokens),
            req.finished,
        )
        if not req.future.done():
            text = final_text if final_text is not None else tokenizer_service.decode(req.generated_tokens)
            req.future.set_result(text)
        if getattr(req, "queue", None) is not None:
            req.queue.put_nowait("[DONE]")
        self._free_block_table(req)

    def _fail_request_timeout(self, req: InferenceRequest) -> None:
        """Fail a request that timed out before generating any tokens."""
        if not req.future.done():
            req.future.set_exception(
                asyncio.TimeoutError("Streaming request timed out before generating any tokens.")
            )
        if getattr(req, "queue", None) is not None:
            req.queue.put_nowait(("[ERROR]", "generation timed out"))
        self._free_block_table(req)

    def _fail_single_request(self, req: InferenceRequest, exc: Exception) -> None:
        """Fail exactly one request after an unrecoverable generation-step error.

        `exc` itself (which may contain internal detail -- stack-trace-flavored
        text, memory sizes, file paths, ...) is kept on the request's `future`
        for internal bookkeeping only. The client-facing SSE message is always
        the generic `INTERNAL_ERROR_MESSAGE`; full detail is already logged
        server-side by the caller. Used by `_retry_requests_individually` for
        a request that fails even in its own batch of one; callers are
        responsible for removing `req` from `self.active_requests` themselves.
        """
        if not req.future.done():
            req.future.set_exception(exc)
        if getattr(req, "queue", None) is not None:
            req.queue.put_nowait(("[ERROR]", INTERNAL_ERROR_MESSAGE))
        self._free_block_table(req)

    def _evict_cancelled_requests(self) -> None:
        """Drop requests whose client has already disconnected.

        No finalization is attempted: the future is already in its cancelled
        terminal state (calling set_result/set_exception on it would raise
        InvalidStateError), and there's no client left to read `req.queue`.
        """
        cancelled = [r for r in self.active_requests if r.future.cancelled()]
        for req in cancelled:
            self._free_block_table(req)
            streaming_metrics.record_cancelled_eviction()
        self.active_requests = [r for r in self.active_requests if not r.future.cancelled()]

    def _free_block_table(self, req: InferenceRequest) -> None:
        """Return a request's paged KV blocks, if it has any.

        Guarded on `block_ids` being non-empty so this never forces the
        (lazily-constructed) paged cache -- and therefore the model -- to
        load just to free a table that was never populated.
        """
        if req.block_table.block_ids:
            self.paged_cache.free(req.block_table)

    def _evict_expired_requests(self) -> None:
        """Evict requests past their deadline.

        A request that already generated some tokens is finished with that
        partial output; one that hasn't is failed with a timeout error.
        """
        now = time.monotonic()
        still_active = []
        for req in self.active_requests:
            if req.deadline is not None and req.deadline <= now:
                streaming_metrics.record_timeout_eviction()
                if req.generated_tokens:
                    self._finish_request(req)
                else:
                    logger.info("Request timed out before generating any tokens: prompt=%s", req.prompt)
                    self._fail_request_timeout(req)
            else:
                still_active.append(req)
        self.active_requests = still_active

    def _forward_and_sample(self, batch_inputs: _BatchInputs, reqs: List[InferenceRequest]):
        """Run the forward pass, repetition penalty, and per-request sampling.

        `reqs` must be index-aligned with `batch_inputs`'s rows -- normally
        `self.active_requests`, but may be a single-request subset during
        isolation retries (see `_retry_requests_individually`).
        """
        logits, new_past = self.engine.forward_step(
            batch_inputs.input_ids,
            batch_inputs.attention_mask,
            batch_inputs.past_key_values,
            position_ids=batch_inputs.position_ids,
            logit_gather_indices=batch_inputs.logit_gather_indices,
        )
        # Penalize repeats against each request's FULL history (prompt +
        # everything generated so far), not `batch_inputs.input_ids` -- that's
        # only this step's new-tokens-this-step slice (a single token during
        # decode), which would make the penalty effectively a no-op after the
        # first step, since `torch.unique()` on one token has nothing to
        # penalize. Each request's real length differs (mixed prefill/decode,
        # different prompt lengths), so this is a ragged list, not a dense
        # tensor -- `apply_repetition_penalty` accepts either.
        full_histories = [req.input_ids[0] for req in reqs]
        logits = self.engine.apply_repetition_penalty(logits, full_histories)
        next_tokens = torch.stack([
            self.engine.sample(
                logits[i].unsqueeze(0),
                req.temperature,
                model_settings.top_k,
                model_settings.top_p,
            )
            for i, req in enumerate(reqs)
        ])
        return next_tokens, new_past

    async def _retry_requests_individually(self) -> None:
        """Isolate a persistent whole-batch failure to whichever request(s) cause it.

        Called after two whole-batch failures in a row (see `_step`). A
        batched forward pass fails or succeeds as a unit, so at that point
        there's no way to attribute the failure to one row from the exception
        alone. Retry every currently-active request in its own batch of
        one: whichever succeed are dispatched normally, same as any other
        step; whichever fail again are failed individually via
        `_fail_single_request` instead of taking every co-batched request
        down with them.
        """
        for req in list(self.active_requests):
            if req not in self.active_requests:
                # Removed by a `_dispatch_tokens` finish or an earlier
                # iteration's failure this loop.
                continue
            single_batch = self._build_batch_inputs([req])
            if single_batch is None:
                continue
            try:
                next_tokens, new_past = self._forward_and_sample(single_batch, [req])
            except Exception as solo_exc:
                logger.exception(
                    "Request failed even in isolation after batch retry failed twice; "
                    "failing it alone: prompt=%s error=%s",
                    req.prompt,
                    solo_exc,
                )
                self._fail_single_request(req, solo_exc)
                self.active_requests = [r for r in self.active_requests if r is not req]
                empty_device_cache(self.engine.device)
                continue
            await self._dispatch_tokens([req], next_tokens, new_past, single_batch.past_width, single_batch.new_lengths)

    async def _step(self):
        """Run a single token generation step for *all* active requests.

        Orchestrates eviction of dead requests, batch preparation, the
        engine's forward pass, penalty application, per-request sampling,
        and token dispatch.
        """
        # 0. Drop requests whose client disconnected or whose deadline has
        #    passed, before doing any further work (scheduler concern).
        self._evict_cancelled_requests()
        self._evict_expired_requests()
        if not self.active_requests:
            return

        step_start = time.monotonic()
        active_count = len(self.active_requests)

        # 1. Prepare batched tensors -- mixing prefill and decode rows
        #    (scheduler concern)
        batch_inputs = self._prepare_batch()
        if batch_inputs is None:
            return

        # 2-4. Forward pass, repetition penalty, sampling (engine concern).
        # Retried once on transient failure before falling back to
        # per-request isolation -- a batched forward pass fails or succeeds
        # as a unit, so there's no per-request granularity to retry at here.
        # The retry is only likely to help for an OOM-shaped failure, so free
        # whatever cached-but-unused device memory PyTorch's allocator is
        # holding onto first -- retrying against the exact same memory state
        # that just failed would almost certainly fail again.
        try:
            next_tokens, new_past = self._forward_and_sample(batch_inputs, self.active_requests)
        except Exception as exc:
            logger.warning("Generation step failed, retrying once: %s", exc)
            empty_device_cache(self.engine.device)
            try:
                next_tokens, new_past = self._forward_and_sample(batch_inputs, self.active_requests)
            except Exception as exc2:
                # Two whole-batch failures in a row: the failure may be
                # specific to one poisoned request (bad cached state, a
                # degenerate shape, ...) rather than every co-batched
                # request. Retry each active request in its own batch of one
                # instead of failing everyone -- see
                # `_retry_requests_individually`.
                logger.exception(
                    "Generation step failed again after retry; isolating requests one at a time: %s", exc2
                )
                await self._retry_requests_individually()
                next_tokens = None

        # 5. Dispatch tokens and update request state (scheduler concern).
        # Skipped if isolation already handled dispatch above (`next_tokens`
        # stays `None` in that branch).
        if next_tokens is not None:
            await self._dispatch_tokens(self.active_requests, next_tokens, new_past, batch_inputs.past_width, batch_inputs.new_lengths)

        # The paged cache's pool never shrinks, but PyTorch's own device
        # allocator caches freed tensor memory rather than returning it to
        # the system. Always release it once there's nothing left to do; while
        # still busy, check actual memory pressure every step (a cheap
        # metadata query) and clear proactively *before* usage ever gets
        # close to the device's ceiling -- not on a fixed step schedule,
        # which could easily be too slow for a single long-running request
        # that never hits an idle gap.
        if not self.active_requests:
            empty_device_cache(self.engine.device)
        else:
            maybe_empty_device_cache(self.engine.device)

        streaming_metrics.record_batch_size(active_count)
        streaming_metrics.record_token_throughput(active_count, time.monotonic() - step_start)
        streaming_metrics.record_batch_occupancy(active_count, self.max_batch_size)
        # Use the already-constructed cache, if any, rather than the
        # `paged_cache` property -- this step may not have touched the
        # paged cache at all (e.g. `_prepare_batch` is mocked out in tests),
        # and this metric isn't worth forcing it to build.
        if self._paged_cache is not None:
            streaming_metrics.record_cache_utilization(
                self._paged_cache.capacity - len(self._paged_cache.free_blocks),
                self._paged_cache.capacity,
            )

    async def run(self):
        """Main scheduler loop.

        Repeatedly adds new requests, executes a token step for all active requests,
        and sleeps briefly when idle.
        """
        while True:
            try:
                await self._add_new_requests()
                if not self.active_requests:
                    await asyncio.sleep(0.01)
                    continue
                await self._step()
            except Exception as exc:
                logger.exception("Scheduler error during run loop: %s", exc)
                await asyncio.sleep(1)
