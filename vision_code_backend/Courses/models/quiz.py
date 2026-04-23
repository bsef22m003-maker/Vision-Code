from django.db import models
from Accounts.models import User
from django.conf import settings

from Courses.models import ContentItem
class Quiz(models.Model):
    content_item = models.OneToOneField(
        ContentItem,
        on_delete=models.CASCADE,
        related_name="quiz"
    )

    total_marks = models.PositiveIntegerField()
    passing_marks = models.PositiveIntegerField()

    class Meta:
        db_table = "quizzes"

class QuizQuestion(models.Model):
    quiz = models.ForeignKey(
        Quiz,
        on_delete=models.CASCADE,
        related_name="questions"
    )

    question_text = models.TextField()
    order = models.PositiveIntegerField()

    class Meta:
        db_table = "quiz_questions"
        ordering = ["order"]

    def __str__(self):
        return self.question_text
class QuizOption(models.Model):
    question = models.ForeignKey(
        QuizQuestion,
        on_delete=models.CASCADE,
        related_name="options"
    )

    option_text = models.CharField(max_length=255)
    is_correct = models.BooleanField(default=False)

    class Meta:
        db_table = "quiz_options"