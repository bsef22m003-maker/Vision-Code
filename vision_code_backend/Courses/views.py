# Courses/views.py

from rest_framework import viewsets, filters
from rest_framework.decorators import action
from rest_framework.permissions import AllowAny, IsAuthenticated
from rest_framework.response import Response
from rest_framework.exceptions import PermissionDenied

from django_filters.rest_framework import DjangoFilterBackend
from django.db.models import Count
from .models import Course, CourseModule, ContentItem
from .serializers.course import (
    CourseListSerializer,
    CourseDetailSerializer
)
from .serializers.module import CourseModuleSerializer
from .serializers.content_item import ContentItemSerializer

from Accounts.permissions import IsInstructorOrAdmin
from Enrollment_Learning.models import Enrollment
from Enrollment_Learning.serializers import EnrollmentSerializer

class CourseViewSet(viewsets.ModelViewSet):
    def get_queryset(self):
        return Course.objects.select_related("instructor").annotate(
            total_students=Count("enrollments", distinct=True),
            total_modules=Count("modules", distinct=True)
        )
    filter_backends = [
        DjangoFilterBackend,
        filters.SearchFilter,
        filters.OrderingFilter,
    ]

    filterset_fields = [
        "difficulty_level",
        "is_published",
        "instructor",   # ✅ filter by instructor ID
    ]

    search_fields = ["title", "topics"]
    ordering_fields = ["created_at", "price"]

    def get_serializer_class(self):
        if self.action == "list":
            return CourseListSerializer
        return CourseDetailSerializer

    def get_permissions(self):
        if self.action in ("create", "update", "partial_update", "destroy"):
            return [IsAuthenticated(), IsInstructorOrAdmin()]
        return [AllowAny()]

    def perform_create(self, serializer):
        serializer.save(instructor=self.request.user)

    @action(detail=True, methods=["post"], permission_classes=[IsAuthenticated])
    def enroll(self, request, pk=None):
        course = self.get_object()
        enrollment, created = Enrollment.objects.get_or_create(
            user=request.user,
            course=course
        )
        serializer = EnrollmentSerializer(enrollment)
        return Response(serializer.data, status=201 if created else 200)
class CourseModuleViewSet(viewsets.ModelViewSet):
        queryset = CourseModule.objects.select_related("course").prefetch_related(
        "content_items__article",
        "content_items__quiz__questions__options",
        "content_items__assignment",
        )
        serializer_class = CourseModuleSerializer

        filter_backends = [
            DjangoFilterBackend,
            filters.SearchFilter,
            filters.OrderingFilter,
        ]

        filterset_fields = ["course"]
        search_fields = ["title"]
        ordering_fields = ["module_order"]

        def get_permissions(self):
            if self.action in ("create", "update", "partial_update", "destroy"):
                return [IsAuthenticated(), IsInstructorOrAdmin()]
            return [AllowAny()]

        def perform_create(self, serializer):
            course = serializer.validated_data["course"]
            user = self.request.user

            if not user.is_staff and course.instructor_id != user.id:
                raise PermissionDenied(
                    "Only the course instructor can add modules."
                )

            serializer.save()


class ContentItemViewSet(viewsets.ModelViewSet):
    queryset = ContentItem.objects.select_related(
    "module",
    "module__course",
    "article",
    "quiz",
    "assignment",
    ).prefetch_related(
    "quiz__questions__options"
    )
    serializer_class = ContentItemSerializer

    filter_backends = [
        DjangoFilterBackend,
        filters.SearchFilter,
        filters.OrderingFilter,
    ]

    filterset_fields = ["module", "content_type"]
    search_fields = ["title"]
    ordering_fields = ["created_at", "estimated_duration_minutes"]

    def get_permissions(self):
        if self.action in ("create", "update", "partial_update", "destroy"):
            return [IsAuthenticated(), IsInstructorOrAdmin()]
        return [AllowAny()]

    def perform_create(self, serializer):
        module = serializer.validated_data["module"]
        user = self.request.user

        if not user.is_staff and module.course.instructor_id != user.id:
            raise PermissionDenied(
                "Only the course instructor can create content."
            )

        serializer.save()

from .models import Article, Quiz, QuizQuestion, QuizOption
from .serializers.article import ArticleSerializer
from .serializers.quiz import (
    QuizSerializer,
    QuizQuestionSerializer,
    QuizQuestionWriteSerializer,
    QuizOptionSerializer,
    QuizOptionWriteSerializer 
)

# -----------------------------
# ARTICLE VIEWSET
# -----------------------------
class ArticleViewSet(viewsets.ModelViewSet):
    queryset = Article.objects.select_related("content_item")
    serializer_class = ArticleSerializer

    def get_permissions(self):
        if self.action in ("create", "update", "partial_update", "destroy"):
            return [IsAuthenticated(), IsInstructorOrAdmin()]
        return [AllowAny()]


# -----------------------------
# QUIZ VIEWSET
# -----------------------------
class QuizViewSet(viewsets.ModelViewSet):
    queryset = Quiz.objects.select_related("content_item").prefetch_related("questions__options")
    serializer_class = QuizSerializer

    def get_permissions(self):
        if self.action in ("create", "update", "partial_update", "destroy"):
            return [IsAuthenticated(), IsInstructorOrAdmin()]
        return [AllowAny()]


# -----------------------------
# QUIZ QUESTION VIEWSET
# -----------------------------
class QuizQuestionViewSet(viewsets.ModelViewSet):
    queryset = QuizQuestion.objects.prefetch_related("options")

    def get_serializer_class(self):
        if self.action in ["create", "update", "partial_update"]:
            return QuizQuestionWriteSerializer
        return QuizQuestionSerializer

    def get_permissions(self):
        if self.action in ("create", "update", "partial_update", "destroy"):
            return [IsAuthenticated(), IsInstructorOrAdmin()]
        return [AllowAny()]


# -----------------------------
# QUIZ OPTION VIEWSET
# -----------------------------
# class QuizOptionWriteSerializer(serializers.ModelSerializer):
#     class Meta:
#         model = QuizOption
#         fields = ("id", "question", "option_text", "is_correct")


class QuizOptionViewSet(viewsets.ModelViewSet):
    queryset = QuizOption.objects.all()

    def get_serializer_class(self):
        if self.action in ["create", "update", "partial_update"]:
            return QuizOptionWriteSerializer   # ✅ write
        return QuizOptionSerializer            # ✅ read

    def get_permissions(self):
        if self.action in ("create", "update", "partial_update", "destroy"):
            return [IsAuthenticated(), IsInstructorOrAdmin()]
        return [AllowAny()]