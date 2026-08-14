# Cache and Inference Engine

## Cache Layer

### `cache/paged_kv_cache.py`

Block-based KV cache storage shared by every request active in the `ContinuousScheduler` (see [Scheduler Layer](Scheduler-Layer#scheduler-layer)). Pure PyTorch indexing -- no fused kernel, so it works identically on MPS/CPU/CUDA. This is a storage/allocation optimization only: it replaces the realloc-and-copy of a monolithically growing per-request tensor with writes into pre-allocated fixed-size blocks. Every step still materializes a dense `(batch, heads, seq, head_dim)` tensor via `gather_dense()` for the model's forward pass -- there is no fused kernel available on MPS to read scattered blocks directly.

`BlockTable` (`@dataclass`) -- one request's own view into the pool:
- `block_ids: list[int]` -- which blocks in the pool belong to this request.
- `length: int` -- real (unpadded) token count stored so far.
- Freed and reset to empty (`block_ids = []`, `length = 0`) by `PagedKVCache.free()`.

`PagedKVCache.__init__(num_layers, num_kv_heads, head_dim, block_size=16, dtype=torch.float32, device="cpu", initial_capacity_blocks=64)`:
- Allocates `key_pool`/`value_pool`: one `(capacity_blocks, num_kv_heads, block_size, head_dim)` tensor per layer, plus `free_blocks: list[int]` tracking unused block indices.
- Shape parameters mirror the loaded model's config -- see `ContinuousScheduler.paged_cache`'s derivation of `num_kv_heads`/`head_dim`/`num_layers`.

`_grow_pool(new_blocks)`:
- Allocates `new_blocks` more per-layer key/value tensors and `torch.cat`s them onto the existing pool; extends `free_blocks` with the new indices. Called by `__init__` and by `_ensure_free`.

`_ensure_free(n_blocks)`:
- If fewer than `n_blocks` are currently free, grows the pool by `max(self.capacity, n_blocks - len(free_blocks))` -- i.e. at least a full doubling, or exactly what's needed if that's larger -- so growth is amortized rather than happening one block at a time.

`allocate(table, n_tokens)`:
- Ensures `table` has enough blocks for `n_tokens` more real tokens (ceil-division block math), pulling from `free_blocks` (growing the pool first via `_ensure_free` if needed).

`append(table, keys_per_layer, values_per_layer)`:
- Appends one step's newly computed K/V for every layer into `table`'s blocks in one call. `keys_per_layer[l]`/`values_per_layer[l]` must have shape `(num_kv_heads, n_new, head_dim)`. Allocates blocks as needed via `allocate()`, then writes token-by-token across block boundaries (a request's new tokens can straddle the end of one block and the start of the next).

`gather_dense(tables) -> (keys_per_layer, values_per_layer, real_lengths)`:
- Materializes a left-padded, batched dense view for the model's forward pass: each `keys_per_layer[l]`/`values_per_layer[l]` has shape `(batch, num_kv_heads, max_len, head_dim)`, left-padded per row to the batch's longest real length (`max_len = max(real_lengths)`). `real_lengths[i]` is request `i`'s true (unpadded) token count -- the metadata the scheduler's prefill/decode mixing needs to build correct attention masks and position ids.
- Per-row gathering (`_gather_row`) walks a table's `block_ids`, concatenating full blocks plus a partial final block for the remainder.

`free(table)`:
- Releases `table`'s blocks back to `free_blocks` and resets it to empty.

`is_valid(table) -> bool`:
- Structural self-check (mirrors the defensive pattern `transformers.DynamicCache` uses internally): every block id in `table.block_ids` must be in-range and not already in `free_blocks`, and `table.length` must be consistent with the number of blocks held (`(n_blocks - 1) * block_size < length <= n_blocks * block_size`). Used by `ContinuousScheduler._prepare_batch()` to detect a corrupted/stale table and fall back to a fresh prefill for that request rather than feeding the model garbage.

---

## Inference Engine

### `engine/generator.py`

This module encapsulates model inference and token-selection logic.

Imports:
- `asyncio`, `Protocol`, `Sequence`
- `torch`
- `model_loader` from `engine.model_loader`
- `tokenizer_service` from `tokenizer.tokenizer_service`
- `find_stop_index` from `utils.stop_sequences`
- `model_settings`, `logging_settings` from `settings.settings`
- `setup_logger` from `logger`

`GenerationRequest` (`typing.Protocol`): the structural interface `generate_batch()` depends on, decoupling the engine from the scheduler package (any object with this shape, e.g. `scheduler.request.InferenceRequest`, can be batched without the engine importing scheduler-owned types). Fields: `future`, `queue`, `temperature`, `max_tokens`, `generated_tokens`, `finished`, `stop_sequences`.

`InferenceEngine.__init__()`:
- `self.device = model_settings.device`.
- `self._model = None` -- deferred; actual retrieval happens lazily via the `model` property, to avoid initializing `torch`/model state at import time (prevents semaphore leaks and is safe with multi-worker servers).

`model` property:
- If `self._model is None`, sets it via `model_loader._get_model()`. Returns `self._model`.

`invalidate_model_cache()`:
- Sets `self._model = None`, so the next `.model` access re-fetches from `model_loader`. Used after `model_loader.reload()` swaps the underlying weights out from under an already-running server (see `scheduler/model_swap.py`).

#### `sample(logits, temperature, top_k=0, top_p=1.0)`

- If `temperature <= 0`, greedy: `torch.argmax(logits, dim=-1)`, shape `(batch, 1)`.
- Otherwise: scales logits by `1/temperature`; applies top-k filtering (threshold via `torch.topk`); applies top-p (nucleus) filtering (cumulative softmax mass, `scatter_`-masked); samples with `torch.multinomial`.

#### `forward_step(input_ids, attention_mask, past_key_values=None, position_ids=None, logit_gather_indices=None)`

- Raises `RuntimeError` if `self.model` is `None`.
- Calls `self.model(input_ids=input_ids, attention_mask=attention_mask, past_key_values=past_key_values, position_ids=position_ids, use_cache=True)`.
- `position_ids`: optional explicit per-token positions, needed when the batch mixes rows with different real past lengths and/or padding (prefill/decode mixing) -- the model's implicit default position handling assumes a uniform past length across the batch, which isn't true in that case. `None` preserves the implicit behavior for pure-decode/pure-prefill batches.
- `logit_gather_indices`: optional per-row column index selecting which position's logits are the real next-token prediction for that row -- needed when a mixed batch's new-tokens region is right-padded, so the real prediction isn't always at column `-1`. If given, gathers `logits[batch_indices, logit_gather_indices, :]`; otherwise falls back to the unconditional `logits[:, -1, :]`.
- Returns `(logits.clone(), outputs.past_key_values)`, logits normalized to shape `(batch, vocab_size)`.

#### `apply_repetition_penalty(logits, input_ids, penalty=None)`

- Defaults `penalty` to `model_settings.repetition_penalty`; no-op if `penalty == 1.0`.
- For each batch row, finds unique tokens already present in `input_ids[i]` and penalizes their logits (multiply if negative, divide if positive) by `penalty`. Mutates `logits` in place and returns it.

#### `eos_token_id` property

Returns `self.model.config.eos_token_id`; raises `RuntimeError` if the model isn't loaded.

#### `generate(input_ids, max_tokens=-1, temperature=-1.0)`

A simple sequential (no KV-cache-across-requests, no eviction/insertion) generation path, used only for rapid prototyping/testing -- **not** used by the continuous scheduler, which uses `forward_step`/`sample`/`apply_repetition_penalty` directly for its paged-cache-aware loop.

#### `generate_batch(input_ids, attention_mask, requests: Sequence[GenerationRequest])`

The engine entry point used by `BatchScheduler`. Tracks active requests as `(original_index, request)` tuples -- `list(enumerate(requests))` -- rather than a wrapper object, so there's no separate state to keep in sync with the underlying request. Runs a `torch.no_grad()` loop while active requests remain:
1. Filters out requests whose `future.cancelled()` is `True`.
2. Forward pass: full batch if `past_key_values is None`, else `next_tokens` with the cached `past_key_values`.
3. Vectorized repetition penalty across the batch.
4. Samples a token per active request, using that request's own `temperature` and the global `model_settings.top_k`/`top_p`.
5. Per request: appends the token to `r.generated_tokens`, streams it to `r.queue` if present, then checks `r.stop_sequences` the same way the streaming path does -- decodes the full `generated_tokens`, calls `find_stop_index()`, and if matched, sets `stop_text` to the pre-match text. A request finishes (removed from the active batch, `outputs[original_idx]` set) when `stop_text is not None`, or EOS is reached, or `max_tokens` is hit; the output is `stop_text` if a stop matched, else the full decode.
6. Appends new tokens to `input_ids`/`attention_mask`; compacts the active batch (and `past_key_values` via `batch_select_indices`) whenever any request finished this step.

After the loop, any request without a set output (should not normally happen) is filled in via a final decode, and every request with a live `queue` receives the `"[DONE]"` sentinel.

Global singleton:
- `engine = InferenceEngine()`
