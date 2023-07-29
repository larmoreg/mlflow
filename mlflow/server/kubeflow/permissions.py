from dataclasses import dataclass


@dataclass
class Permission:
    name: str
    can_read: bool
    can_update: bool
    can_delete: bool


READ = Permission(
    name="READ",
    can_read=True,
    can_update=False,
    can_delete=False,
)

EDIT = Permission(
    name="EDIT",
    can_read=True,
    can_update=True,
    can_delete=False,
)

NO_PERMISSIONS = Permission(
    name="NO_PERMISSIONS",
    can_read=False,
    can_update=False,
    can_delete=False,
)

ALL_PERMISSIONS = {
    READ.name: READ,
    EDIT.name: EDIT,
    NO_PERMISSIONS.name: NO_PERMISSIONS,
}


def get_permission(permission: str) -> Permission:
    return ALL_PERMISSIONS[permission]
