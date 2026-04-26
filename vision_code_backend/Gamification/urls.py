# Gamification/urls.py
from rest_framework.routers import DefaultRouter
from .views import BadgeViewSet, UserBadgeViewSet

router = DefaultRouter()
router.register(r"badges", BadgeViewSet, basename="badge")
router.register(r"my-badges", UserBadgeViewSet, basename="my-badges")

urlpatterns = router.urls
