# Gamification/views.py
from rest_framework import viewsets, permissions
from rest_framework.response import Response

from .models import Badge, UserBadge
from .serializers import BadgeSerializer, UserBadgeSerializer
from .permissions import IsAdminOrInstructor


# --------------------------------------------------
# BADGES (MASTER LIST)
# --------------------------------------------------
class BadgeViewSet(viewsets.ModelViewSet):
    queryset = Badge.objects.all()
    serializer_class = BadgeSerializer

    def get_permissions(self):
        if self.request.method in permissions.SAFE_METHODS:
            return [permissions.AllowAny()]
        return [permissions.IsAuthenticated(), IsAdminOrInstructor()]


# --------------------------------------------------
# USER BADGES (MY ACHIEVEMENTS)
# --------------------------------------------------
class UserBadgeViewSet(viewsets.ReadOnlyModelViewSet):
    serializer_class = UserBadgeSerializer
    permission_classes = [permissions.IsAuthenticated]

    def get_queryset(self):
        return UserBadge.objects.filter(user=self.request.user)
