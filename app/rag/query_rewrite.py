from __future__ import annotations

import json
import logging

from openai import AsyncOpenAI

logger = logging.getLogger(__name__)


class QueryRewriter:
    """多轮指代/口语化查询改写。

    关键约束：保留原问题一路召回，改写失败或改写漂移时不至于完全丢失意图。
    """

    def __init__(self, base_url: str, api_key: str, model: str) -> None:
        self.client = AsyncOpenAI(base_url=base_url, api_key=api_key)
        self.model = model

    async def rewrite(self, question: str, history: list[dict[str, str]] | None = None) -> list[str]:
        history = history or []
        messages = [
            {
                "role": "system",
                "content": (
                    "你是企业知识库检索查询改写器。把用户问题改写成 1-3 条适合检索的中文查询。"
                    "要求：保留数字、编号、商品编码、专有名词；补全多轮指代；不要回答问题。"
                    '只输出 JSON：{"queries":["..."]}'
                ),
            },
            *history[-6:],
            {"role": "user", "content": question},
        ]
        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                temperature=0.0,
                messages=messages,
                response_format={"type": "json_object"},
            )
            raw = response.choices[0].message.content or "{}"
            fence = chr(96) * 3
            raw = raw.strip()
            if raw.startswith(fence + "json"):
                raw = raw[len(fence + "json") :]
            if raw.endswith(fence):
                raw = raw[: -len(fence)]
            data = json.loads(raw.strip())
            queries = [str(item).strip() for item in data.get("queries", []) if str(item).strip()]
            queries = [query for query in queries if len(query) <= 200][:3]
            return list(dict.fromkeys([question, *queries]))
        except Exception as exc:  # noqa: BLE001
            logger.warning("query rewrite failed: %s", exc)
            return [question]
