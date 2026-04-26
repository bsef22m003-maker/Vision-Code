

# Accounts/views.py
from rest_framework import viewsets, generics
from rest_framework.decorators import api_view, permission_classes
from rest_framework.permissions import AllowAny, IsAuthenticated
from rest_framework.response import Response

from django.contrib.auth import get_user_model
from .serializers import UserSerializer, UserProfileSerializer
from .auth_serializers import SignupSerializer

User = get_user_model()


class SignupView(generics.CreateAPIView):
    serializer_class = SignupSerializer
    permission_classes = [AllowAny]


@api_view(["POST"])
@permission_classes([AllowAny])
def login_and_get_tokens(request):
    """
    Optional placeholder — we use SimpleJWT endpoints for token obtain/refresh:
    POST /api/token/ and POST /api/token/refresh/
    """
    return Response({"detail": "Use /api/token/ to obtain tokens"}, status=400)


class UserViewSet(viewsets.ReadOnlyModelViewSet):
    """
    Admin-use or public user listing (read-only).
    """
    queryset = User.objects.all().order_by("id")
    serializer_class = UserSerializer


class UserMeView(generics.RetrieveUpdateAPIView):
    """
    GET: retrieve current user (including profile)
    PATCH: update fields on user and nested profile (partial)
    """
    serializer_class = UserSerializer
    permission_classes = [IsAuthenticated]

    def get_object(self):
        return self.request.user

    def patch(self, request, *args, **kwargs):
        user = self.get_object()
        # handle profile updates if present
        profile_data = request.data.get("profile", None)
        if profile_data:
            profile_serializer = UserProfileSerializer(user.profile, data=profile_data, partial=True, context={"request": request})
            profile_serializer.is_valid(raise_exception=True)
            profile_serializer.save()
        # update user fields (first_name, last_name)
        return super().partial_update(request, *args, **kwargs)
