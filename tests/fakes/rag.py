from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class FakeDoc:
    page_content: str
    metadata: dict = field(default_factory=dict)


class FakeAsyncRetriever:
    def __init__(self, docs: list[FakeDoc]) -> None:
        self.docs = docs

    async def ainvoke(self, _query: str):
        return self.docs
