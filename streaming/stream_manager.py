import asyncio
from typing import AsyncGenerator, Union

from scheduler.request import InferenceRequest
from tokenizer.tokenizer_service import tokenizer_service
from utils.stop_sequences import find_stop_index
from settings.settings import logging_settings
from logger import setup_logger

logger = setup_logger(__name__, level=logging_settings.log_level, log_file=logging_settings.log_file)


async def stream_response(req: InferenceRequest) -> AsyncGenerator[Union[str, dict], None]:
    """Yield decoded tokens from an ``InferenceRequest``'s streaming queue.

    The generator reads token IDs from ``req.queue`` until the sentinel ``"[DONE]"``
    is received, accumulating token IDs and decoding the entire sequence to yield the
    new text delta. This handles multi-byte character boundaries correctly.

    A ``("[ERROR]", message)`` tuple sentinel (pushed when a request is evicted
    for timing out or failing server-side) yields a single SSE ``error`` event
    and ends the stream instead of ``[DONE]``.
    """
    tokens = []
    yielded_text = ""

    while True:
        token = await req.queue.get()
        logger.debug("Stream manager received token=%s for prompt=%s", token, req.prompt)
        if token == "[DONE]":
            logger.debug("Stream manager received DONE for prompt=%s", req.prompt)
            break
        if isinstance(token, tuple) and len(token) == 2 and token[0] == "[ERROR]":
            logger.debug("Stream manager received ERROR=%s for prompt=%s", token[1], req.prompt)
            yield {"event": "error", "data": token[1]}
            break

        # Load tokenizer lazily and skip any special token emissions.
        if tokenizer_service.tokenizer is None:
            tokenizer_service.load()

        special_ids = getattr(tokenizer_service.tokenizer, "all_special_ids", None)
        if special_ids is not None and token in special_ids:
            logger.debug("Stream manager skipping special token=%s for prompt=%s", token, req.prompt)
            continue

        tokens.append(token)
        current_text = tokenizer_service.decode(tokens)

        # Strip trailing unicode replacement character (incomplete UTF-8 sequence)
        clean_text = current_text
        while clean_text.endswith("\ufffd"):
            clean_text = clean_text[:-1]

        # Checked against the full decoded text (not just this token's delta)
        # before any buffering decision below, so the stop sequence -- and
        # anything after it -- is never flushed to the client.
        if req.stop_sequences:
            stop_idx = find_stop_index(clean_text, req.stop_sequences)
            if stop_idx is not None:
                if stop_idx > len(yielded_text):
                    trimmed = clean_text[len(yielded_text):stop_idx]
                    if trimmed:
                        yield trimmed
                logger.debug("Stream manager hit stop sequence for prompt=%s", req.prompt)
                return

        if len(clean_text) <= len(yielded_text):
            continue

        delta = clean_text[len(yielded_text):]
        should_emit = delta.endswith(" ") or delta[-1] in ".,;:!?" or len(delta) >= 16

        if should_emit:
            yielded_text += delta
            logger.debug(
                "Stream manager decoded tokens -> delta '%s' for prompt=%s",
                delta,
                req.prompt,
            )
            yield delta
        else:
            logger.debug(
                "Stream manager buffering partial delta '%s' for prompt=%s",
                delta,
                req.prompt,
            )
