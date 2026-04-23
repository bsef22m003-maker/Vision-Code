# Courses/serializers/course.py
from rest_framework import serializers
from Courses.models import Course
from .module import CourseModuleSerializer


class CourseListSerializer(serializers.ModelSerializer):
    instructor = serializers.SerializerMethodField()
    total_students = serializers.IntegerField(read_only=True)
    total_modules = serializers.IntegerField(read_only=True)
    class Meta:
        model = Course
        fields = (
            "id",
            "title",
            "description",
            "difficulty_level",
            "topics",
            "is_published",
            "price",
            "instructor",
            "total_students",
            "total_modules",
        )

    def get_instructor(self, obj):
        return {
            "id": obj.instructor.id,
            "name": f"{obj.instructor.first_name} {obj.instructor.last_name}",
        }


class CourseDetailSerializer(serializers.ModelSerializer):
    instructor = serializers.SerializerMethodField()
    modules = CourseModuleSerializer(many=True, read_only=True)
    total_students = serializers.IntegerField(read_only=True)
    total_modules = serializers.IntegerField(read_only=True)
    class Meta:
        model = Course
        fields = (
            "id",
            "title",
            "description",
            "difficulty_level",
            "topics",
            "is_published",
            "price",
            "instructor",
            "total_students",
            "total_modules",
            "modules",
        )

    def get_instructor(self, obj):
        return {
            "id": obj.instructor.id,
            "email": obj.instructor.email,
            "name": f"{obj.instructor.first_name} {obj.instructor.last_name}",
        }