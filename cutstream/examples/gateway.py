"""Adapt decoded chat-completion content deltas to a streaming text gateway."""

import codecs
from collections.abc import AsyncIterator

from cutstream import Plan


async def visible_text(completion, plan: Plan) -> AsyncIterator[str]:
    """Use inside the SDK's async stream context, with one completion choice.

    The producer supplies append-only content deltas with provider-side stops
    disabled. The caller chooses how to handle separate tool-call events.
    """
    async def chunks():
        async for event in completion:
            if event.choices:
                text = event.choices[0].delta.content
                if text:
                    yield text.encode("utf-8")

    decoder = codecs.getincrementaldecoder("utf-8")()
    async for chunk in plan.afilter(chunks()):
        text = decoder.decode(chunk)
        if text:
            yield text
    tail = decoder.decode(b"", final=True)
    if tail:
        yield tail
