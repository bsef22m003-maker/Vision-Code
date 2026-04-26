# Courses/serializers/module.py
from rest_framework import serializers
from Courses.models import CourseModule
from .content_item import ContentItemSerializer


class CourseModuleSerializer(serializers.ModelSerializer):
    content_items = ContentItemSerializer(many=True, read_only=True)

    class Meta:
        model = CourseModule
        fields = (
            "id",
            "course",
            "title",
            "description",
            "module_order",
            "learning_objectives",
            "content_items",
        )