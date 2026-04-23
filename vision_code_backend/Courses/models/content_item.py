from django.db import models
from Accounts.models import User
from django.conf import settings
from Courses.models import CourseModule
class ContentItem(models.Model):
    CONTENT_TYPE_CHOICES = [
        ("article", "Article"),
        ("quiz", "Quiz"),
        ("coding_problem", "Coding Problem"),
        ("assignment", "Assignment"),
    ]

    DIFFICULTY_CHOICES = [
        ("easy", "Easy"),
        ("medium", "Medium"),
        ("hard", "Hard"),
    ]

    module = models.ForeignKey(
        CourseModule,
        on_delete=models.CASCADE,
        related_name="content_items"
    )

    content_type = models.CharField(
        max_length=20,
        choices=CONTENT_TYPE_CHOICES
    )

    title = models.CharField(max_length=200)

    difficulty = models.CharField(
        max_length=20,
        choices=DIFFICULTY_CHOICES
    )

    estimated_duration_minutes = models.PositiveIntegerField()

    order = models.PositiveIntegerField()  # ordering inside module

    is_published = models.BooleanField(default=False)

    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        db_table = "content_items"
        ordering = ["order"]

    def __str__(self):
        return f"{self.title} ({self.content_type})"