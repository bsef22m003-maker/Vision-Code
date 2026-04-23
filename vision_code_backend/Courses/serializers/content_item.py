# Courses/serializers/content_item.py
from rest_framework import serializers
from Courses.models import ContentItem
from .article import ArticleSerializer
from .quiz import QuizSerializer
from .coding_problem import CodingProblemSerializer
from .assignment import AssignmentSerializer


class ContentItemSerializer(serializers.ModelSerializer):
    content_data = serializers.SerializerMethodField()
    class Meta:
        model = ContentItem
        fields = (
            "id",
            "module",
            "content_type",
            "difficulty",
            "estimated_duration_minutes",
            "order",
            "content_data",
            
        )

    def get_content_data(self, obj):
        if obj.content_type == "article" and hasattr(obj, "article"):
            return ArticleSerializer(obj.article).data

        if obj.content_type == "quiz" and hasattr(obj, "quiz"):
            return QuizSerializer(obj.quiz).data

        if obj.content_type == "coding_problem" and hasattr(obj, "coding_problem"):
            return CodingProblemSerializer(obj.coding_problem).data

        if obj.content_type == "assignment" and hasattr(obj, "assignment"):
            return AssignmentSerializer(obj.assignment).data

        return None