from app.api.schemas import UserContext
from app.services.permission_service import PermissionService


def test_permission_filter_contains_tenant_and_acl():
    user = UserContext(
        user_id="u1",
        tenant_id="t1",
        roles=["employee"],
        department_ids=["d1"],
    )
    service = PermissionService()
    milvus_filter = service.milvus_filter(user)
    es_filters = service.es_filters(user)

    assert 'tenant_id == "t1"' in milvus_filter
    assert "visibility" in milvus_filter
    assert any("tenant_id" in str(item) for item in es_filters)


def test_admin_bypasses_document_acl():
    admin = UserContext(user_id="admin", tenant_id="t1", roles=["admin"])
    service = PermissionService()
    assert service.can_manage_document(admin, owner_user_id="someone", department_id="d2")
