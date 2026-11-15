import asyncio
import json

from app.core.config import get_settings
from app.services.container import build_container


async def main() -> None:
    settings = get_settings()
    container = await build_container(settings)
    try:
        result = await container.evaluation.run("app/evaluation/golden_set.jsonl", user=container.demo_user, top_k=5)
        print(json.dumps(result, ensure_ascii=False, indent=2))
    finally:
        await container.close()


if __name__ == "__main__":
    asyncio.run(main())
