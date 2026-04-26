# Courses/serializers/quiz.py
from rest_framework import serializers
from Courses.models import Quiz, QuizQuestion, QuizOption

class QuizOptionSerializer(serializers.ModelSerializer):
    class Meta:
        model = QuizOption
        fields = ("id", "option_text", "is_correct")
class QuizOptionWriteSerializer(serializers.ModelSerializer):
    class Meta:
        model = QuizOption
        fields = ("id", "question", "option_text", "is_correct")

class QuizQuestionWriteSerializer(serializers.ModelSerializer):
    class Meta:
        model = QuizQuestion
        fields = ("id", "quiz", "question_text", "order")

class QuizQuestionSerializer(serializers.ModelSerializer):
    options = QuizOptionSerializer(many=True, read_only=True)

    class Meta:
        model = QuizQuestion
        fields = ("id", "question_text", "order", "options")
    

from Courses.models import ContentItem

class QuizSerializer(serializers.ModelSerializer):
    content_item = serializers.PrimaryKeyRelatedField(
        queryset=ContentItem.objects.all()
    )

    questions = QuizQuestionSerializer(many=True, read_only=True)

    class Meta:
        model = Quiz
        fields = ("id", "content_item", "total_marks", "passing_marks", "questions")