# Courses/serializers/coding_problem.py
from rest_framework import serializers
from Courses.models import CodingProblem, CodingTestCase


class CodingTestCaseSerializer(serializers.ModelSerializer):
    class Meta:
        model = CodingTestCase
        fields = ("id", "input_data", "expected_output")


class CodingProblemSerializer(serializers.ModelSerializer):
    test_cases = CodingTestCaseSerializer(many=True, read_only=True)

    class Meta:
        model = CodingProblem
        fields = (
            "id",
            "content_item",
            "problem_statement",
            "starter_code",
            "test_cases",
        )