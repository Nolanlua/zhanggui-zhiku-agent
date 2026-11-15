from __future__ import annotations

import json

from app.api.schemas import UserContext


def quote(value: str) -> str:
    return json.dumps(value, ensure_ascii=False)[1:-1]


class PermissionService:
    """把用户身份翻译成检索层的 ACL 过滤条件。

    关键原则：权限必须在召回前过滤，而不是生成后删引用。
    """

    def milvus_filter(self, user: UserContext) -> str:
        conditions = [f'tenant_id == "{quote(user.tenant_id)}"']
        if "admin" in user.roles:
            return " and ".join(conditions)
        acl_parts = ['visibility == "public"']
        if user.department_ids:
            departments = ", ".join(f'"{quote(item)}"' for item in user.department_ids)
            acl_parts.append(f'(visibility == "department" and department_id in [{departments}])')
        acl_parts.append(f'(visibility == "private" and owner_user_id == "{quote(user.user_id)}")')
        conditions.append("(" + " or ".join(acl_parts) + ")")
        return " and ".join(conditions)

    def es_filters(self, user: UserContext) -> list[dict]:
        filters: list[dict] = [{"term": {"tenant_id": user.tenant_id}}]
        if "admin" in user.roles:
            return filters
        should: list[dict] = [{"term": {"visibility": "public"}}]
        if user.department_ids:
            should.append(
                {
                    "bool": {
                        "must": [
                            {"term": {"visibility": "department"}},
                            {"terms": {"department_id": user.department_ids}},
                        ]
                    }
                }
            )
        should.append(
            {
                "bool": {
                    "must": [
                        {"term": {"visibility": "private"}},
                        {"term": {"owner_user_id": user.user_id}},
                    ]
                }
            }
        )
        filters.append({"bool": {"should": should, "minimum_should_match": 1}})
        return filters

    def can_manage_document(self, user: UserContext, owner_user_id: str | None, department_id: str | None) -> bool:
        if "admin" in user.roles or owner_user_id == user.user_id:
            return True
        return bool(department_id and department_id in user.department_ids)
