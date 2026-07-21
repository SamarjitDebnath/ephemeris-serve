import torch
from engine.model_loader import model_loader
from tokenizer.tokenizer_service import tokenizer_service
from settings.settings import model_settings, logging_settings
from logger import setup_logger

logger = setup_logger(__name__, level=logging_settings.log_level, log_file=logging_settings.log_file)


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

    def forward_step(self, input_ids, attention_mask, past_key_values=None):
        """Run a single forward pass through the model.

        Args:
            input_ids: Token IDs tensor ``(batch, seq_len)``.
            attention_mask: Attention mask tensor ``(batch, seq_len)``.
            past_key_values: Optional KV cache from a previous step.

        Returns:
            Tuple of ``(logits, new_past_key_values)`` where *logits* has
            shape ``(batch, vocab_size)`` (last-position only).
        """
        if self.model is None:
            raise RuntimeError("Model failed to load")

        logger.debug(
            "Forward step input shapes: input_ids=%s attention_mask=%s past_key_values=%s",
            tuple(input_ids.shape),
            tuple(attention_mask.shape) if attention_mask is not None else None,
            type(past_key_values).__name__ if past_key_values is not None else None,
        )
        
        logger.info("Model device: %s, Model dtype: %s", self.model.device, self.model.dtype)

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=True,
        )
        logits = outputs.logits

        # # Squeeze out any unexpected leading dimensions beyond (batch, seq, vocab)
        # while logits.dim() > 3:
        #     logits = logits.squeeze(1)
        if logits.dim() == 3:
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
            input_ids: Full input IDs tensor ``(batch, seq_len)`` used to
                determine which tokens have already appeared.
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

        for i in range(input_ids.shape[0]):
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
    async def generate_batch(self, input_ids, attention_mask, requests):
        from scheduler.request import ActiveRequest
        
        batch_size = input_ids.shape[0]
        if batch_size == 0:
            return [""] * len(requests)

        # Move tensors to correct device
        input_ids = input_ids.to(self.device)
        attention_mask = attention_mask.to(self.device)

        logger.info(f"Starting batched generation for {batch_size} requests.")

        # Create active requests wrappers
        active_requests = [ActiveRequest(r, i) for i, r in enumerate(requests)]
        
        # Pre-allocate output list for all original requests
        outputs = [None] * len(requests)

        with torch.no_grad():
            past_key_values = None
            next_tokens = None
            
            while len(active_requests) > 0:
                active_requests = [r for r in active_requests if not r.request.future.cancelled()]
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
                
                for i, r in enumerate(active_requests):
                    token = self.sample(
                        logits[i].unsqueeze(0),
                        r.temperature,
                        model_settings.top_k,
                        model_settings.top_p
                    )
                    next_tokens[i] = token
                
                # 4. Update request states and check for finished conditions
                keep_indices = []
                for i, r in enumerate(active_requests):
                    token_id = next_tokens[i].item()
                    r.generated_tokens.append(token_id)
                    
                    if getattr(r.request, "queue", None) is not None:
                        await r.request.queue.put(token_id)
                    
                    if token_id == self.model.config.eos_token_id or len(r.generated_tokens) >= r.max_tokens:
                        r.finished = True
                        outputs[r.original_index] = tokenizer_service.decode(r.generated_tokens)
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