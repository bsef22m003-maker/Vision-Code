from django.contrib import admin
from .models import Badge, UserBadge


@admin.register(Badge)
class BadgeAdmin(admin.ModelAdmin):
    list_display = (
        "id",
        "badge_name",
        "badge_type",
    )
    list_filter = ("badge_type",)
    search_fields = ("badge_name",)


@admin.register(UserBadge)
class UserBadgeAdmin(admin.ModelAdmin):
    list_display = (
        "id",
        "user",
        "badge",
        "earned_at",
    )
    list_filter = ("badge__badge_type",)
    search_fields = ("user__email", "badge__badge_name")
