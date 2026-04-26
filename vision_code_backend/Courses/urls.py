# Courses/urls.py
from rest_framework.routers import DefaultRouter
from .views import (
    CourseViewSet,
    CourseModuleViewSet,
    ContentItemViewSet,
    ArticleViewSet,
    QuizViewSet,
    QuizQuestionViewSet,
    QuizOptionViewSet,
)

router = DefaultRouter()

router.register(r"courses", CourseViewSet, basename="course")
router.register(r"course-modules", CourseModuleViewSet, basename="course-modules")
router.register(r"content-items", ContentItemViewSet, basename="content-items")

# ✅ NEW
router.register(r"articles", ArticleViewSet, basename="articles")
router.register(r"quizzes", QuizViewSet, basename="quizzes")
router.register(r"quiz-questions", QuizQuestionViewSet, basename="quiz-questions")
router.register(r"quiz-options", QuizOptionViewSet, basename="quiz-options")

urlpatterns = router.urls