# Accounts/urls.py
from django.urls import path, include
from rest_framework.routers import DefaultRouter
from .views import UserViewSet, SignupView, UserMeView
from rest_framework_simplejwt.views import  TokenRefreshView
from .token_views import CustomTokenObtainPairView
router = DefaultRouter()
router.register(r"users", UserViewSet, basename="user")

urlpatterns = [
    path("auth/signup/", SignupView.as_view(), name="signup"),
    #the token is for the simple jwt default view
    #path("token/", TokenObtainPairView.as_view(), name="token_obtain_pair"),
    #but for this we should use the custom view with custom serializer, with role in token
    path("token/", CustomTokenObtainPairView.as_view(), name="token_obtain_pair"),

    path("token/refresh/", TokenRefreshView.as_view(), name="token_refresh"),
    path("", include(router.urls)),
    path("users/me/", UserMeView.as_view(), name="user-me"),
]
