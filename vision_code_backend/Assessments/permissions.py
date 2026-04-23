# Assessments/permissions.py
from rest_framework.permissions import BasePermission, SAFE_METHODS


class IsInstructor(BasePermission):
    def has_permission(self, request, view):
        return request.user.is_authenticated and request.user.role == "instructor"


class IsSubmissionOwner(BasePermission):
    def has_object_permission(self, request, view, obj):
        return obj.user == request.user
