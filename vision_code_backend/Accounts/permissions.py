
from rest_framework import permissions

class IsInstructorOrAdmin(permissions.BasePermission):
    """
    Allow only users with role 'instructor' or staff/admin.
    """
    def has_permission(self, request, view):
        user = request.user
        if not user or not user.is_authenticated:
            return False
        if getattr(user, "role", None) == "instructor" or user.is_staff:
            return True
        return False
