from django.db import models
from Accounts.models import User
from django.conf import settings
from Courses.models import ContentItem
class CodingProblem(models.Model):
    content_item = models.OneToOneField(
        ContentItem,
        on_delete=models.CASCADE,
        related_name="coding_problem"
    )

    problem_statement = models.TextField()
    constraints = models.TextField(blank=True)
    starter_code = models.JSONField(default=dict)
    time_limit_seconds = models.PositiveIntegerField(default=1)

    class Meta:
        db_table = "coding_problems"

class CodingTestCase(models.Model):
    problem = models.ForeignKey(
        CodingProblem,
        on_delete=models.CASCADE,
        related_name="test_cases"
    )

    input_data = models.TextField()
    expected_output = models.TextField()
    is_hidden = models.BooleanField(default=True)

    class Meta:
        db_table = "coding_test_cases"