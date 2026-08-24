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
    is received, decoding incrementally to yield each new text delta. Multi-byte
    character boundaries are handled correctly: a token that completes only part
    of a character is held until the rest arrives.

    Decoding is incremental rather than whole-history because ``decode`` is
    linear in token count, so re-decoding everything per token would make a
    stream quadratic in its own length. Two offsets bound the work: ``read_offset``
    marks what has already been turned into text, ``prefix_offset`` trails it far
    enough that the decode still has the context it needs to be correct at the
    seam. ``decode`` is not decomposable -- ``decode(a + b)`` is not
    ``decode(a) + decode(b)`` for byte-level BPE -- so the delta is taken as the
    difference between two overlapping decodes rather than by decoding the new
    token alone.

    A ``("[ERROR]", message)`` tuple sentinel (pushed when a request is evicted
    for timing out or failing server-side) yields a single SSE ``error`` event
    and ends the stream instead of ``[DONE]``.
    """
    tokens = []
    yielded_text = ""
    # Full text decoded so far, accumulated from deltas. Equivalent to
    # decoding the whole token list every step, without paying for it.
    decoded_text = ""
    # Incremental detokenization offsets -- see the docstring.
    prefix_offset = 0
    read_offset = 0

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

        # Two overlapping decodes; their difference is the new text. A trailing
        # replacement character means this token only completed part of a
        # character, so hold everything and let the next token finish it --
        # this replaces the old strip-trailing-\ufffd loop.
        prefix_text = tokenizer_service.decode(tokens[prefix_offset:read_offset])
        new_text = tokenizer_service.decode(tokens[prefix_offset:])
        if len(new_text) <= len(prefix_text) or new_text.endswith("\ufffd"):
            logger.debug(
                "Stream manager holding incomplete character for prompt=%s", req.prompt
            )
            continue

        new_delta = new_text[len(prefix_text):]
        decoded_text += new_delta
        prefix_offset = read_offset
        read_offset = len(tokens)

        # Checked against the accumulated text (not just this token's delta)
        # before any buffering decision below, so the stop sequence -- and
        # anything after it -- is never flushed to the client.
        #
        # Only the region a *new* match could occupy is searched: a sequence
        # ending inside `new_delta` starts at most `max_stop_length - 1`
        # characters before it, and anything earlier would already have
        # matched on a previous token and returned.
        if req.stop_sequences:
            search_from = max(0, len(decoded_text) - len(new_delta) - req.max_stop_length + 1)
            relative_idx = find_stop_index(decoded_text[search_from:], req.stop_sequences)
            if relative_idx is not None:
                stop_idx = search_from + relative_idx
                if stop_idx > len(yielded_text):
                    trimmed = decoded_text[len(yielded_text):stop_idx]
                    if trimmed:
                        yield trimmed
                logger.debug("Stream manager hit stop sequence for prompt=%s", req.prompt)
                return

        if len(decoded_text) <= len(yielded_text):
            continue

        delta = decoded_text[len(yielded_text):]
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
