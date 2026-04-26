from django.contrib import admin
from .models import Enrollment, UserProgress, LearningPath


@admin.register(Enrollment)
class EnrollmentAdmin(admin.ModelAdmin):
    list_display = (
        "id",
        "user",
        "course",
        "enrollment_status",
        "progress_percentage",
        "enrolled_at",
        "completed_at",
    )
    list_filter = ("enrollment_status", "course")
    search_fields = ("user__email", "course__title")


@admin.register(UserProgress)
class UserProgressAdmin(admin.ModelAdmin):
    list_display = (
        "id",
        "user",
        "content",
        "status",
        "time_spent_minutes",
        "attempts_count",
        "started_at",
        "completed_at",
    )
    list_filter = ("status",)
    search_fields = ("user__email", "content__title")


@admin.register(LearningPath)
class LearningPathAdmin(admin.ModelAdmin):
    list_display = (
        "id",
        "user",
        "course",
        "ai_model",
        "generated_at",
        "is_active",
    )
    list_filter = ("is_active", "ai_model", "course")
    search_fields = ("user__email", "course__title")
