from django.db import models
from Accounts.models import User
from django.conf import settings
from Courses.models import ContentItem
class Article(models.Model):
    content_item = models.OneToOneField(
        ContentItem,
        on_delete=models.CASCADE,
        related_name="article"
    )

    body = models.TextField()

    summary = models.TextField(blank=True)

    class Meta:
        db_table = "articles"