# Assessments/urls.py
from rest_framework.routers import DefaultRouter
from .views import CodingProblemViewSet, SubmissionViewSet

router = DefaultRouter()
router.register(r"problems", CodingProblemViewSet, basename="problem")
router.register(r"submissions", SubmissionViewSet, basename="submission")

urlpatterns = router.urls
