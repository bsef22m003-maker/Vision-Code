from django.db import models
from Courses.models import ContentItem


class Assignment(models.Model):
    content_item = models.OneToOneField(
        ContentItem,
        on_delete=models.CASCADE,
        related_name="assignment"
    )

    instructions = models.TextField()
    max_score = models.PositiveIntegerField()

    due_date = models.DateTimeField(null=True, blank=True)

    class Meta:
        db_table = "assignments"