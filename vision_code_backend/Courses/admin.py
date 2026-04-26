from django.contrib import admin
from .models import (
    Course,
    CourseModule,
    ContentItem,
    Article,
    Quiz,
    QuizQuestion,
    QuizOption,
    CodingProblem,
    CodingTestCase,
    Assignment,
)

# ----------------------------
# COURSE & MODULE
# ----------------------------
@admin.register(Course)
class CourseAdmin(admin.ModelAdmin):
    list_display = ("id", "title", "instructor", "difficulty_level", "is_published")
    list_filter = ("difficulty_level", "is_published")
    search_fields = ("title",)


@admin.register(CourseModule)
class CourseModuleAdmin(admin.ModelAdmin):
    list_display = ("id", "course", "title", "module_order")
    ordering = ("module_order",)


# ----------------------------
# CONTENT ITEM (BASE)
# ----------------------------
@admin.register(ContentItem)
class ContentItemAdmin(admin.ModelAdmin):
    list_display = ("id", "title", "content_type", "module", "order")
    list_filter = ("content_type",)
    ordering = ("order",)


# ----------------------------
# ARTICLE
# ----------------------------
@admin.register(Article)
class ArticleAdmin(admin.ModelAdmin):
    list_display = ("id", "content_item")


# ----------------------------
# QUIZ
# ----------------------------
class QuizOptionInline(admin.TabularInline):
    model = QuizOption
    extra = 2


class QuizQuestionInline(admin.TabularInline):
    model = QuizQuestion
    extra = 1


@admin.register(Quiz)
class QuizAdmin(admin.ModelAdmin):
    list_display = ("id", "content_item")
    inlines = [QuizQuestionInline]


@admin.register(QuizQuestion)
class QuizQuestionAdmin(admin.ModelAdmin):
    list_display = ("id", "quiz", "question_text")
    inlines = [QuizOptionInline]


@admin.register(QuizOption)
class QuizOptionAdmin(admin.ModelAdmin):
    list_display = ("id", "question","option_text", "is_correct")


# ----------------------------
# CODING PROBLEM
# ----------------------------
@admin.register(CodingProblem)
class CodingProblemAdmin(admin.ModelAdmin):
    list_display = ("id", "content_item")


@admin.register(CodingTestCase)
class CodingTestCaseAdmin(admin.ModelAdmin):
    list_display = ("id", "problem", "input_data", "expected_output")


# ----------------------------
# ASSIGNMENT
# ----------------------------
@admin.register(Assignment)
class AssignmentAdmin(admin.ModelAdmin):
    list_display = ("id", "content_item", "due_date")