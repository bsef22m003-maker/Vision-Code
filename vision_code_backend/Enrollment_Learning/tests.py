# Enrollment_Learning/tests.py
from django.test import TestCase
from rest_framework.test import APIClient
from Accounts.models import User
from Courses.models import Course
from Enrollment_Learning.models import Enrollment


class EnrollmentAPITests(TestCase):
    def setUp(self):
        self.client = APIClient()
        self.user = User.objects.create_user(
            email="test@example.com", password="pass123", role="student"
        )
        self.client.force_authenticate(self.user)

        self.course = Course.objects.create(
            title="Test Course",
            description="Demo",
            instructor=self.user,
            difficulty_level="beginner",
            price=0.0,
        )

    def test_enrollment_create(self):
        response = self.client.post("/api/enrollments/", {"course": self.course.id})
        self.assertEqual(response.status_code, 201)
