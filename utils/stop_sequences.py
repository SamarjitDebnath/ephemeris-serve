from typing import Sequence


def find_stop_index(text: str, stop_sequences: Sequence[str]) -> int | None:
    """Return the index in `text` where the earliest-occurring stop sequence
    begins, or `None` if none of `stop_sequences` appear in `text`.

    Used to both decide when generation should halt and to truncate output
    so the stop sequence itself (and anything after it) is never returned.
    """
    earliest = None
    for stop in stop_sequences:
        if not stop:
            continue
        idx = text.find(stop)
        if idx != -1 and (earliest is None or idx < earliest):
            earliest = idx
    return earliest
