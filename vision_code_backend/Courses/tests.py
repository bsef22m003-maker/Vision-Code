# courses/tests.py
from rest_framework.test import APITestCase
from django.urls import reverse
from django.contrib.auth import get_user_model
from Courses.models import Course

User = get_user_model()

class CourseTests(APITestCase):
    def setUp(self):
        self.instructor = User.objects.create_user(email="inst@example.com", password="pass12345", role="instructor")
        self.student = User.objects.create_user(email="stu@example.com", password="pass12345", role="student")

    def test_course_create_and_list(self):
        self.client.force_authenticate(user=self.instructor)
        url = reverse("course-list")
        data = {"title":"Test Course","description":"desc","course_type":"instructor_published","difficulty_level":"beginner","topics":["arrays"], "is_published": True, "price": "0.00"}
        r = self.client.post(url, data, format="json")
        self.assertEqual(r.status_code, 201)
        self.client.force_authenticate(user=None)
        r2 = self.client.get(url)
        self.assertEqual(r2.status_code, 200)
        self.assertTrue(any(c["title"]=="Test Course" for c in r2.data["results"]))
