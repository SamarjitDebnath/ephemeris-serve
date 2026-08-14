# Model and Tokenizer

## Model and Tokenizer

### `engine/model_loader.py`

Loads, warms up, and (at runtime) hot-swaps the Hugging Face language model.

Imports:
- `gc`
- `torch`
- `AutoModelForCausalLM` from `transformers`
- `model_settings` from `settings.settings`
- `tokenizer_service` from `tokenizer.tokenizer_service`
- `empty_device_cache` from `utils.device_cache` (see [Utility Helpers](Reference#utility-helpers))

Class `ModelLoader`:
- `self.model` is initialized as `None`.
- `load()`: if `self.model is None`, builds and assigns it via `self._build_model(model_settings.model_name)`.
- `_build_model(model_name)`: the shared build routine --
  - Selects dtype based on device: `torch.float16` for `mps`, `torch.float32` otherwise.
  - Calls `AutoModelForCausalLM.from_pretrained(model_name, dtype=dtype)`, moves it to `model_settings.device`, calls `.eval()`.
  - Logs the outcome (device, dtype, model type, vocab size) and returns the model object -- it does **not** touch `self.model`.
  - Used for both the initial `load()` and runtime `reload()`.
- `reload(model_name)`: runtime model hot-swap, used by `scheduler/model_swap.py` once it has confirmed no requests are in flight --
  - Calls `_build_model(model_name)` fully *before* touching `self.model`. If it raises (bad repo id, network error, OOM, ...), the previously-loaded model is left running untouched.
  - On success: swaps `self.model` to the new model, updates `model_settings.model_name`, then drops the old model reference, runs `gc.collect()`, and calls `utils.device_cache.empty_device_cache(model_settings.device)` to release its memory. This was previously a private module-level `_empty_device_cache()` helper duplicated in a couple of places; it's now the single shared implementation in `utils/device_cache.py`, also used by `scheduler/continuous_scheduler.py`, `scheduler/batch_scheduler.py`, and `scheduler/model_swap.py`.
- `_get_model()`: lazily calls `load()` if needed and returns the model instance.
- `warmup()`: obtains the model instance, encodes `"Warmup request"` via `tokenizer_service.encode(..., return_tensors=True)`, moves `input_ids` to the model device, and runs one `torch.no_grad()` forward pass (`torch.argmax` on the resulting logits) to exercise the forward path once before serving real traffic.

Global singleton:
- `model_loader = ModelLoader()`

### `tokenizer/tokenizer_service.py`

Manages tokenizer initialization, encoding, decoding, and (at runtime) hot-swapping.

Imports:
- `AutoTokenizer` from `transformers`
- `model_settings` from `settings.settings`

Class `TokenizerService`:
- `self.tokenizer` is initialized as `None`.
- `load()`: if `self.tokenizer is None`, builds and assigns it via `self._build_tokenizer(model_settings.model_name)`.
- `_build_tokenizer(model_name)`: the shared build routine -- instantiates `AutoTokenizer.from_pretrained(model_name)`, sets `pad_token = eos_token` if missing, sets `padding_side = "left"`, and returns the tokenizer object without touching `self.tokenizer`.
- `reload(model_name)`: mirrors `ModelLoader.reload` -- builds the new tokenizer fully via `_build_tokenizer` before publishing it to `self.tokenizer`, so a failure leaves the previous tokenizer active.

Encoding:
- `encode(text, return_tensors=False)` loads the tokenizer if needed.
- If `return_tensors=True`, returns the raw transformer output dictionary containing `input_ids` and `attention_mask`.
- Otherwise, returns a plain Python list of token IDs from the first batch element.
- Uses truncation and `max_length=model_settings.max_length` to constrain sequence length.

Decoding:
- `decode(tokens)` loads the tokenizer if needed and returns decoded text with `skip_special_tokens=True`.

Global singleton:
- `tokenizer_service = TokenizerService()`

Low-level data shapes:
- `input_ids` returned by `tokenizer(..., return_tensors='pt')` is a tensor of shape `(1, seq_len)`.
- `attention_mask` is a tensor of shape `(1, seq_len)`.
