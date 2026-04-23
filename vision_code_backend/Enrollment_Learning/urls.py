from django.urls import path
from rest_framework.routers import DefaultRouter

from .views import (
    EnrollmentViewSet,
    MyCoursesView,
    UserProgressViewSet,
    LearningPathViewSet,
)

router = DefaultRouter()

# ✅ FIXED
router.register(r"", EnrollmentViewSet, basename="enrollments")
#router.register(r"progress", UserProgressViewSet)
#router.register(r"learning-paths", LearningPathViewSet)

urlpatterns = [
    path("my-courses/", MyCoursesView.as_view(), name="my-courses"),
]

urlpatterns += router.urls