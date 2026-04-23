from django.contrib import admin
from .models import AIModel, CVSession


@admin.register(AIModel)
class AIModelAdmin(admin.ModelAdmin):
    list_display = (
        "id",
        "model_name",
        "model_type",
        "version",
        "is_active",
        "created_at",
    )
    list_filter = ("model_type", "is_active")
    search_fields = ("model_name", "version")


@admin.register(CVSession)
class CVSessionAdmin(admin.ModelAdmin):
    list_display = (
        "id",
        "user",
        "content",
        "session_start",
        "session_end",
        "engagement_score",
    )
    list_filter = ("content",)
    search_fields = ("user__email", "content__title")


