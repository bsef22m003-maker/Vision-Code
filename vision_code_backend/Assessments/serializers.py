# Assessments/serializers.py
from rest_framework import serializers
from .models import CodingProblem, Submission, TestCaseResult
from Courses.models import ContentItem
from Enrollment_Learning.models import Enrollment, UserProgress
from django.utils import timezone


# --------------------------------------------------
# CODING PROBLEM
# --------------------------------------------------
class CodingProblemSerializer(serializers.ModelSerializer):
    content_title = serializers.CharField(source="content.title", read_only=True)

    class Meta:
        model = CodingProblem
        fields = "__all__"


# --------------------------------------------------
# SUBMISSION
# --------------------------------------------------
class SubmissionSerializer(serializers.ModelSerializer):
    user = serializers.HiddenField(default=serializers.CurrentUserDefault())

    class Meta:
        model = Submission
        fields = "__all__"
        read_only_fields = (
            "passed_all_tests",
            "score",
            "submitted_at",
            "execution_results",
            "plagiarism_report",
        )

    def validate(self, attrs):
        user = self.context["request"].user
        problem = attrs["problem"]

        # find course via content
        content = problem.content
        course = content.module.course

        # ensure enrollment
        if not Enrollment.objects.filter(user=user, course=course).exists():
            raise serializers.ValidationError(
                "You must be enrolled in the course to submit this problem."
            )

        return attrs

    def create(self, validated_data):
        submission = super().create(validated_data)

        # ---- SIMULATED JUDGE (MVP) ----
        # Later: move this to Celery / worker
        passed = True
        submission.passed_all_tests = passed
        submission.score = 100 if passed else 0
        submission.execution_results = {"status": "simulated"}
        submission.save()

        # ---- UPDATE USER PROGRESS ----
        content = submission.problem.content
        UserProgress.objects.update_or_create(
            user=submission.user,
            content=content,
            defaults={
                "status": "completed",
                "completed_at": timezone.now(),
            },
        )

        return submission


# --------------------------------------------------
# TEST CASE RESULT
# --------------------------------------------------
class TestCaseResultSerializer(serializers.ModelSerializer):
    class Meta:
        model = TestCaseResult
        fields = "__all__"
