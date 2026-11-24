"""Adapt decoded chat-completion content deltas to a streaming text gateway."""

import codecs
from collections.abc import AsyncIterator

from cutstream import Plan


async def visible_text(completion, plan: Plan, *, choice_index: int = 0) -> AsyncIterator[str]:
    """Use inside the SDK's async stream context, to emit one selected choice.

    The producer supplies append-only content deltas with provider-side stops
    disabled. Select by the SDK's choice index across chunks, including chunks
    with multiple choices. Closing the context also closes any unread choices.
    The caller chooses how to handle separate tool-call events.
    """
    async def chunks():
        async for event in completion:
            for choice in event.choices:
                if getattr(choice, "index", 0) != choice_index:
                    continue
                text = choice.delta.content
                if text:
                    yield text.encode("utf-8")
                if getattr(choice, "finish_reason", None) is not None:
                    return

    decoder = codecs.getincrementaldecoder("utf-8")()
    async for chunk in plan.afilter(chunks()):
        text = decoder.decode(chunk)
        if text:
            yield text
    tail = decoder.decode(b"", final=True)
    if tail:
        yield tail
