from typing import TYPE_CHECKING, Annotated

from fastapi import Header, HTTPException, Request

from app.api.schemas import UserContext

if TYPE_CHECKING:
    from app.services.container import Container


def get_container(request: Request) -> "Container":
    container = getattr(request.app.state, "container", None)
    if container is None:
        raise HTTPException(status_code=503, detail="service is starting")
    return container


async def get_user(
    x_user_id: Annotated[str, Header()] = "demo-user",
    x_tenant_id: Annotated[str, Header()] = "tenant_demo",
    x_roles: Annotated[str, Header()] = "employee",
    x_department_ids: Annotated[str, Header()] = "",
    x_store_ids: Annotated[str, Header()] = "",
) -> UserContext:
    """开发期从 Header 读取身份；生产环境替换为 JWT/SSO 网关解析。"""

    if not x_user_id or not x_tenant_id:
        raise HTTPException(status_code=401, detail="missing identity")
    split = lambda value: [item.strip() for item in value.split(",") if item.strip()]
    return UserContext(
        user_id=x_user_id,
        tenant_id=x_tenant_id,
        roles=split(x_roles),
        department_ids=split(x_department_ids),
        store_ids=split(x_store_ids),
    )
