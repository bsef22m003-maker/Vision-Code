from django.urls import reverse
from rest_framework.test import APITestCase
from django.contrib.auth import get_user_model

User = get_user_model()

class AuthTests(APITestCase):
    def test_signup_and_token(self):
        url = reverse("signup")
        data = {"email": "testuser@example.com", "password": "testpass123", "first_name":"T"}
        r = self.client.post(url, data, format="json")
        self.assertEqual(r.status_code, 201)
        # obtain token
        url_token = reverse("token_obtain_pair")
        r2 = self.client.post(url_token, {"email":"testuser@example.com","password":"testpass123"}, format="json")
        self.assertEqual(r2.status_code, 200)
        self.assertIn("access", r2.data)
