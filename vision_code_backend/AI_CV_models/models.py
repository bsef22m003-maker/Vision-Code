from django.db import models
from django.conf import settings
from Courses.models import ContentItem

class AIModel(models.Model):
    MODEL_TYPE_CHOICES = [
        ('adaptive_learning', 'Adaptive Learning'),
        ('nlp', 'NLP'),
        ('computer_vision', 'Computer Vision'),
        ('recommendation', 'Recommendation'),
    ]
    
    model_name = models.CharField(max_length=100)
    model_type = models.CharField(max_length=30, choices=MODEL_TYPE_CHOICES)
    version = models.CharField(max_length=50)
    model_parameters = models.JSONField(default=dict)
    is_active = models.BooleanField(default=True)
    created_at = models.DateTimeField(auto_now_add=True)
    
    class Meta:
        db_table = 'Ai_models'



class CVSession(models.Model):
    user = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE, related_name='cv_sessions')
    content = models.ForeignKey(ContentItem, on_delete=models.CASCADE, related_name='cv_sessions')
    session_start = models.DateTimeField()
    session_end = models.DateTimeField(null=True, blank=True)
    attention_metrics = models.JSONField(default=dict)
    emotion_data = models.JSONField(default=dict)
    engagement_score = models.FloatField(default=0.0)
    
    class Meta:
        db_table = 'cv_sessions'
