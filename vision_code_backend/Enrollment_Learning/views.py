
# Enrollment_Learning/views.py
from urllib import response

from rest_framework import viewsets, permissions
from rest_framework.exceptions import PermissionDenied
from .models import Enrollment, UserProgress, LearningPath
from .serializers import (
    EnrollmentSerializer,
    UserProgressSerializer,
    LearningPathSerializer,
)
from django.utils import timezone


# ---------------------------------------------------------
# ENROLLMENT VIEWSET
# ---------------------------------------------------------


class EnrollmentViewSet(viewsets.ModelViewSet):
    permission_classes = [permissions.IsAuthenticated]
    serializer_class = EnrollmentSerializer

    def get_queryset(self):
        user = self.request.user
        if user.is_staff:
            return Enrollment.objects.all().select_related("user", "course")
        return Enrollment.objects.filter(user=user).select_related("user", "course")

    def perform_create(self, serializer):
        serializer.save(user=self.request.user)

# ---------------------------------------------------------
# USER PROGRESS VIEWSET
# ---------------------------------------------------------
class UserProgressViewSet(viewsets.ModelViewSet):
    permission_classes = [permissions.IsAuthenticated]
    serializer_class = UserProgressSerializer

    def get_queryset(self):
        user = self.request.user
        return (
            UserProgress.objects.filter(user=user)
            .select_related("user", "content", "content__module")
        )

    def perform_create(self, serializer):
        serializer.save(user=self.request.user)

    def perform_update(self, serializer):
        serializer.save(user=self.request.user)


# ---------------------------------------------------------
# LEARNING PATH VIEWSET (minimal API)
# ---------------------------------------------------------
class LearningPathViewSet(viewsets.ModelViewSet):
    permission_classes = [permissions.IsAuthenticated]
    serializer_class = LearningPathSerializer

    def get_queryset(self):
        user = self.request.user
        return LearningPath.objects.filter(user=user).select_related(
            "user", "course", "ai_model"
        )

    def perform_create(self, serializer):
        serializer.save(user=self.request.user)


from rest_framework.views import APIView
from rest_framework.permissions import IsAuthenticated
from django.db.models import Count
from Courses.models import Course
from Courses.serializers.course import CourseListSerializer
from rest_framework.response import Response

class MyCoursesView(APIView):
    permission_classes = [IsAuthenticated]

    def get(self, request):
        courses = Course.objects.filter(
            enrollments__user=request.user
        ).select_related("instructor").annotate(
            total_students=Count("enrollments", distinct=True),
            total_modules=Count("modules", distinct=True)
        )

        serializer = CourseListSerializer(courses, many=True)
        return Response(serializer.data)