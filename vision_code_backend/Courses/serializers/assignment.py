# Courses/serializers/assignment.py
from rest_framework import serializers
from Courses.models import Assignment


class AssignmentSerializer(serializers.ModelSerializer):
    class Meta:
        model = Assignment
        fields = "__all__"