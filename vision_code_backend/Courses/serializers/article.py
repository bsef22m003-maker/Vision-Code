# Courses/serializers/article.py
from rest_framework import serializers
from Courses.models import Article


class ArticleSerializer(serializers.ModelSerializer):
    class Meta:
        model = Article
        fields = "__all__"