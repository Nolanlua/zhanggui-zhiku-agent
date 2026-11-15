from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any, Awaitable, Callable

import httpx
from openai import AsyncOpenAI

from app.agent.prompts import TOOL_PLANNER_SYSTEM
from app.api.schemas import UserContext

logger = logging.getLogger(__name__)

ToolHandler = Callable[[dict[str, Any], UserContext], Awaitable[dict[str, Any]]]


@dataclass(slots=True)
class ToolDefinition:
    name: str
    description: str
    parameters: dict[str, Any]
    handler: ToolHandler
    required_roles: tuple[str, ...] = ()


class ToolPermissionError(PermissionError):
    pass


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, ToolDefinition] = {}

    def register(self, tool: ToolDefinition) -> None:
        self._tools[tool.name] = tool

    def schemas(self) -> list[dict[str, Any]]:
        return [
            {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.parameters,
                },
            }
            for tool in self._tools.values()
        ]

    async def execute(self, name: str, arguments: dict[str, Any], user: UserContext) -> dict[str, Any]:
        tool = self._tools.get(name)
        if tool is None:
            return {"ok": False, "error": f"unknown tool: {name}"}
        if tool.required_roles and not (set(tool.required_roles) & set(user.roles)) and "admin" not in user.roles:
            return {"ok": False, "error": "permission denied"}
        try:
            data = await tool.handler(arguments, user)
            return {"ok": True, "data": data}
        except Exception as exc:  # noqa: BLE001
            logger.warning("tool %s failed: %s", name, exc)
            return {"ok": False, "error": str(exc)[:500]}


class InternalApiClient:
    """内部商品/库存/报表服务客户端；生产可替换为 gRPC。"""

    def __init__(self, base_url: str, token: str = "", timeout: float = 3.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.token = token
        self.timeout = timeout

    async def get(self, path: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        headers = {"Authorization": f"Bearer {self.token}"} if self.token else {}
        async with httpx.AsyncClient(base_url=self.base_url, timeout=self.timeout) as client:
            response = await client.get(path, params=params, headers=headers)
            response.raise_for_status()
            return response.json()


def build_default_registry(client: InternalApiClient) -> ToolRegistry:
    registry = ToolRegistry()

    async def get_product_info(arguments: dict[str, Any], user: UserContext) -> dict[str, Any]:
        return await client.get(f"/api/products/{arguments['product_code']}")

    async def get_inventory(arguments: dict[str, Any], user: UserContext) -> dict[str, Any]:
        return await client.get(
            "/api/inventory",
            {"store_id": arguments["store_id"], "sku": arguments["sku"]},
        )

    async def get_sales_report(arguments: dict[str, Any], user: UserContext) -> dict[str, Any]:
        return await client.get(
            "/api/reports/sales",
            {"store_id": arguments["store_id"], "start_date": arguments["start_date"], "end_date": arguments["end_date"]},
        )

    registry.register(
        ToolDefinition(
            name="get_product_info",
            description="查询商品基础信息：名称、规格、编码、保质期。",
            parameters={
                "type": "object",
                "properties": {"product_code": {"type": "string", "description": "商品编码/SKU"}},
                "required": ["product_code"],
            },
            handler=get_product_info,
        )
    )
    registry.register(
        ToolDefinition(
            name="get_inventory",
            description="查询某门店某 SKU 的现有库存、在途库存和可售库存。",
            parameters={
                "type": "object",
                "properties": {
                    "store_id": {"type": "string"},
                    "sku": {"type": "string"},
                },
                "required": ["store_id", "sku"],
            },
            handler=get_inventory,
        )
    )
    registry.register(
        ToolDefinition(
            name="get_sales_report",
            description="查询某门店某时间段的销售报表，用于经营分析。",
            parameters={
                "type": "object",
                "properties": {
                    "store_id": {"type": "string"},
                    "start_date": {"type": "string", "description": "YYYY-MM-DD"},
                    "end_date": {"type": "string", "description": "YYYY-MM-DD"},
                },
                "required": ["store_id", "start_date", "end_date"],
            },
            handler=get_sales_report,
            required_roles=("manager", "admin"),
        )
    )
    return registry


class ToolPlanner:
    """Function Calling 规划器：模型决定调哪个工具，应用负责校验和执行。"""

    def __init__(self, client: AsyncOpenAI, model: str) -> None:
        self.client = client
        self.model = model

    async def plan(
        self,
        question: str,
        user: UserContext,
        registry: ToolRegistry,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        tools = registry.schemas()
        if not tools:
            return [], []
        response = await self.client.chat.completions.create(
            model=self.model,
            temperature=0.0,
            messages=[
                {"role": "system", "content": TOOL_PLANNER_SYSTEM},
                {"role": "user", "content": question},
            ],
            tools=tools,
            tool_choice="auto",
        )
        message = response.choices[0].message
        calls = message.tool_calls or []
        executed: list[dict[str, Any]] = []
        for call in calls[:2]:
            name = call.function.name
            try:
                arguments = json.loads(call.function.arguments or "{}")
            except json.JSONDecodeError:
                arguments = {}
            result = await registry.execute(name, arguments, user)
            executed.append({"name": name, "arguments": arguments, "result": result})
        return [{"name": item["name"], "arguments": item["arguments"]} for item in executed], executed


def tool_results_to_prompt(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return results
