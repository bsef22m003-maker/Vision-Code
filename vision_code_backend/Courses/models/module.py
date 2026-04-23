from django.db import models
from Accounts.models import User
from django.conf import settings

from Courses.models import Course
class CourseModule(models.Model):
    course = models.ForeignKey(Course, on_delete=models.CASCADE, related_name='modules')
    title = models.CharField(max_length=200)
    description = models.TextField(blank=True)
    module_order = models.IntegerField(default=0)
    learning_objectives = models.JSONField(default=list)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    class Meta:
        db_table = 'course_modules'
    def __str__(self):
        return f"{self.course.title} - {self.title}"