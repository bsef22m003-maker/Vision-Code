

# Enrollment_Learning/serializers.py
from datetime import timezone
from rest_framework import serializers

from Courses.models import Course
from .models import Enrollment, UserProgress, LearningPath
from Courses.serializers.course import CourseListSerializer
from Courses.models import ContentItem


# -----------------------------------------------------------
# ENROLLMENT SERIALIZER
# -----------------------------------------------------------
class EnrollmentSerializer(serializers.ModelSerializer):
    user = serializers.SerializerMethodField(read_only=True)

    # accept course id from POST, but still allow nested course display
    course = serializers.PrimaryKeyRelatedField(
        queryset=Course.objects.all(),
        write_only=True
    )
    course_detail = CourseListSerializer(source="course", read_only=True)

    class Meta:
        model = Enrollment
        fields = (
            "id",
            "user",
            "course",
            "course_detail",
            "enrollment_status",
            "enrolled_at",
            "completed_at",
            "progress_percentage",
        )
        read_only_fields = (
            "user",
            "enrolled_at",
            "completed_at",
            "progress_percentage",
        )

    def get_user(self, obj):
        return {"id": obj.user.id, "email": obj.user.email}


# -----------------------------------------------------------
# USER PROGRESS SERIALIZER
# -----------------------------------------------------------
class UserProgressSerializer(serializers.ModelSerializer):
    user = serializers.SerializerMethodField()
    content_title = serializers.CharField(source="content.title", read_only=True)

    class Meta:
        model = UserProgress
        fields = (
            "id",
            "user",
            "content",
            "content_title",
            "status",
            "time_spent_minutes",
            "attempts_count",
            "started_at",
            "completed_at",
            "performance_metrics",
        )
        read_only_fields = ("user", "started_at", "completed_at")

    def get_user(self, obj):
        return {"id": obj.user.id, "email": obj.user.email}

    def validate(self, attrs):
        status = attrs.get("status")
        # Auto-set timestamps
        if status == "in_progress" and not self.instance:
            attrs["started_at"] = timezone.now()
        if status == "completed":
            attrs["completed_at"] = timezone.now()
        return attrs


# -----------------------------------------------------------
# LEARNING PATH SERIALIZER (simple version)
# -----------------------------------------------------------
class LearningPathSerializer(serializers.ModelSerializer):
    user = serializers.SerializerMethodField()
    course = CourseListSerializer(read_only=True)

    class Meta:
        model = LearningPath
        fields = (
            "id",
            "user",
            "course",
            "recommended_sequence",
            "ai_model",
            "generated_at",
            "is_active",
        )

    def get_user(self, obj):
        return {"id": obj.user.id, "email": obj.user.email}
