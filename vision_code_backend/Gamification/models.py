from django.db import models
from django.conf import settings

class Badge(models.Model):
    BADGE_TYPE_CHOICES = [
        ('achievement', 'Achievement'),
        ('skill', 'Skill'),
        ('participation', 'Participation'),
    ]
    
    badge_name = models.CharField(max_length=100)
    description = models.TextField()
    icon_url = models.URLField()
    badge_type = models.CharField(max_length=20, choices=BADGE_TYPE_CHOICES)
    unlock_conditions = models.JSONField(default=dict)
    
    class Meta:
        db_table = 'badges'

class UserBadge(models.Model):
    user = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE, related_name='user_badges')
    badge = models.ForeignKey(Badge, on_delete=models.CASCADE, related_name='user_badges')
    earned_at = models.DateTimeField(auto_now_add=True)
    achievement_context = models.JSONField(default=dict)
    
    class Meta:
        db_table = 'user_badges'
        unique_together = ['user', 'badge']