# Assessments/views.py
from rest_framework import viewsets, permissions
from .models import CodingProblem, Submission, TestCaseResult
from .serializers import (
    CodingProblemSerializer,
    SubmissionSerializer,
    TestCaseResultSerializer,
)
from .permissions import IsInstructor, IsSubmissionOwner


# --------------------------------------------------
# CODING PROBLEMS
# --------------------------------------------------
class CodingProblemViewSet(viewsets.ModelViewSet):
    queryset = CodingProblem.objects.select_related("content")
    serializer_class = CodingProblemSerializer

    def get_permissions(self):
        if self.request.method in permissions.SAFE_METHODS:
            return [permissions.AllowAny()]
        return [permissions.IsAuthenticated(), IsInstructor()]


# --------------------------------------------------
# SUBMISSIONS
# --------------------------------------------------
class SubmissionViewSet(viewsets.ModelViewSet):
    serializer_class = SubmissionSerializer
    permission_classes = [permissions.IsAuthenticated, IsSubmissionOwner]

    def get_queryset(self):
        return Submission.objects.filter(user=self.request.user)

    def perform_create(self, serializer):
        serializer.save(user=self.request.user)


# --------------------------------------------------
# TEST CASE RESULTS (READ ONLY)
# --------------------------------------------------
class TestCaseResultViewSet(viewsets.ReadOnlyModelViewSet):
    serializer_class = TestCaseResultSerializer
    permission_classes = [permissions.IsAuthenticated]

    def get_queryset(self):
        submission_id = self.kwargs["submission_id"]
        return TestCaseResult.objects.filter(
            submission__id=submission_id,
            submission__user=self.request.user,
        )
