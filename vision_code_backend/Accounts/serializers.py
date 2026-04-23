
from rest_framework import serializers
from django.contrib.auth import get_user_model
from .models import UserProfile

User = get_user_model()


class UserProfileSerializer(serializers.ModelSerializer):
    class Meta:
        model = UserProfile
        fields = ("bio", "avatar_url", "timezone", "learning_preferences", "profile_created_at")
        read_only_fields = ("profile_created_at",)


class UserSerializer(serializers.ModelSerializer):
    profile = UserProfileSerializer(read_only=True)

    class Meta:
        model = User
        # fields exposed to clients
        fields = ("id", "email", "first_name", "last_name", "role", "is_active", "created_at", "profile")
        read_only_fields = ("is_active", "created_at", "role")
