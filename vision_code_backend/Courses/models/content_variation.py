from django.db import models
from Accounts.models import User
from django.conf import settings

from Courses.models import ContentItem
class ContentVariation(models.Model):
    VARIATION_TYPE_CHOICES = [
        ('simplified', 'Simplified'),
        ('standard', 'Standard'),
        ('advanced', 'Advanced'),
    ]
    
    content = models.ForeignKey(ContentItem, on_delete=models.CASCADE, related_name='variations')
    variation_type = models.CharField(max_length=20, choices=VARIATION_TYPE_CHOICES)
    adapted_content = models.TextField()
    ai_model = models.ForeignKey('AI_CV_models.AIModel', on_delete=models.SET_NULL, null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    
    class Meta:
        db_table = 'content_variations'