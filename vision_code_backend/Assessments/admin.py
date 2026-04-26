from django.contrib import admin
from .models import Submission, TestCaseResult


@admin.register(Submission)
class SubmissionAdmin(admin.ModelAdmin):
    list_display = ("id", "user", "problem", "passed_all_tests", "score", "submitted_at")
    list_filter = ("passed_all_tests",)
    search_fields = ("user__email",)


@admin.register(TestCaseResult)
class TestCaseResultAdmin(admin.ModelAdmin):
    list_display = ("id", "submission", "test_case_index", "passed")