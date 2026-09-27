import { useState } from "react";
import { useParams, Link, useNavigate } from "react-router-dom";
import { useQuery } from "@tanstack/react-query";
import {
  ArrowLeft,
  BookOpen,
  Code,
  Star,
  Clock,
  CheckCircle,
  ChevronDown,
  ChevronUp,
  Play,
  FileText,
  Users,
  BarChart2,
  Layers,
  Tag,
  LogOut,
  Settings,
  Edit,
  Trash2,
  X,
  Plus,
  HelpCircle,
} from "lucide-react";

import { useAuth } from "../context/FakeAuth";
import { useCourseDetail } from "../hooks/useCourseDetail";
import { getCourseModules } from "../services/courseDetail";
import InstructorCourseDetailSkeleton from "../components/InstructorCourseDetailSkeleton";

import { API_URL } from "../constants";
// ─── Constants ────────────────────────────────────────────────────────────────

const levelGradients = {
  beginner: "from-green-400 to-emerald-500",
  intermediate: "from-purple-500 to-indigo-600",
  advanced: "from-orange-400 to-red-500",
  "all levels": "from-blue-500 to-cyan-600",
};

const levelIcons = {
  beginner: "📚",
  intermediate: "🚀",
  advanced: "⚡",
  "all levels": "🎯",
};

const contentTypeConfig = {
  article: {
    icon: BookOpen,
    color: "text-blue-500",
    badge: "bg-blue-50 text-blue-700 border-blue-200",
    label: "Article",
  },
  quiz: {
    icon: HelpCircle,
    color: "text-purple-500",
    badge: "bg-purple-50 text-purple-700 border-purple-200",
    label: "Quiz",
  },
  exercise: {
    icon: Code,
    color: "text-orange-500",
    badge: "bg-orange-50 text-orange-700 border-orange-200",
    label: "Exercise",
  },
  video: {
    icon: Play,
    color: "text-green-500",
    badge: "bg-green-50 text-green-700 border-green-200",
    label: "Video",
  },
};

// ─── Add Module Modal ────────────────────────────────────────────────────────

function AddModuleModal({ courseId, isOpen, onClose, onSuccess, token }) {
  const [formData, setFormData] = useState({
    title: "",
    description: "",
    learning_objectives: [""],
  });
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState("");

  const handleInputChange = (e) => {
    const { name, value } = e.target;
    setFormData((prev) => ({
      ...prev,
      [name]: value,
    }));
  };

  const handleObjectiveChange = (index, value) => {
    const newObjectives = [...formData.learning_objectives];
    newObjectives[index] = value;
    setFormData((prev) => ({
      ...prev,
      learning_objectives: newObjectives,
    }));
  };

  const addObjective = () => {
    setFormData((prev) => ({
      ...prev,
      learning_objectives: [...prev.learning_objectives, ""],
    }));
  };

  const removeObjective = (index) => {
    setFormData((prev) => ({
      ...prev,
      learning_objectives: prev.learning_objectives.filter(
        (_, i) => i !== index,
      ),
    }));
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setError("");

    if (!formData.title.trim()) {
      setError("Module title is required");
      return;
    }

    try {
      setIsLoading(true);

      const modulesRes = await fetch(
        `${API_URL}/api/courses/course-modules/?course=${courseId}`,
        {
          headers: {
            Authorization: `Bearer ${token}`,
          },
        },
      );

      const modules = await modulesRes.json();
      const moduleOrder = (modules.results?.length || 0) + 1;

      const modulePayload = {
        course: courseId,
        title: formData.title,
        description: formData.description,
        module_order: moduleOrder,
        learning_objectives: formData.learning_objectives
          .filter((obj) => obj.trim())
          .map((obj) => obj.trim()),
      };

      const response = await fetch(`${API_URL}/api/courses/course-modules/`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          Authorization: `Bearer ${token}`,
        },
        body: JSON.stringify(modulePayload),
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(
          errorData.detail || `Failed to create module: ${response.status}`,
        );
      }

      onSuccess();
      setFormData({
        title: "",
        description: "",
        learning_objectives: [""],
      });
      onClose();
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to create module");
    } finally {
      setIsLoading(false);
    }
  };

  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50">
      <div className="bg-white rounded-2xl p-6 w-full max-w-md max-h-96 overflow-y-auto">
        <div className="flex items-center justify-between mb-4">
          <h2 className="text-xl font-bold text-gray-900">Add New Module</h2>
          <button
            onClick={onClose}
            className="p-1 text-gray-400 hover:text-gray-600 transition"
          >
            <X className="w-5 h-5" />
          </button>
        </div>

        {error && (
          <div className="mb-4 p-3 bg-red-50 border border-red-200 rounded-lg text-sm text-red-700">
            {error}
          </div>
        )}

        <form onSubmit={handleSubmit} className="space-y-4">
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">
              Module Title *
            </label>
            <input
              type="text"
              name="title"
              value={formData.title}
              onChange={handleInputChange}
              className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent outline-none"
              placeholder="e.g., Introduction to React"
            />
          </div>

          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">
              Description
            </label>
            <textarea
              name="description"
              value={formData.description}
              onChange={handleInputChange}
              className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent outline-none resize-none"
              rows="3"
              placeholder="Module description..."
            />
          </div>

          <div>
            <div className="flex items-center justify-between mb-2">
              <label className="block text-sm font-medium text-gray-700">
                Learning Objectives
              </label>
              <button
                type="button"
                onClick={addObjective}
                className="text-xs text-blue-600 hover:text-blue-700 font-semibold"
              >
                + Add
              </button>
            </div>
            <div className="space-y-2">
              {formData.learning_objectives.map((obj, idx) => (
                <div key={idx} className="flex gap-2">
                  <input
                    type="text"
                    value={obj}
                    onChange={(e) => handleObjectiveChange(idx, e.target.value)}
                    className="flex-1 px-3 py-2 border border-gray-300 rounded-lg text-sm focus:ring-2 focus:ring-blue-500 focus:border-transparent outline-none"
                    placeholder={`Objective ${idx + 1}`}
                  />
                  {formData.learning_objectives.length > 1 && (
                    <button
                      type="button"
                      onClick={() => removeObjective(idx)}
                      className="px-2 py-2 text-red-500 hover:bg-red-50 rounded-lg transition"
                    >
                      <X className="w-4 h-4" />
                    </button>
                  )}
                </div>
              ))}
            </div>
          </div>

          <div className="flex gap-3 pt-4">
            <button
              type="button"
              onClick={onClose}
              className="flex-1 py-2 px-4 border border-gray-300 text-gray-900 rounded-lg font-semibold hover:bg-gray-50 transition"
              disabled={isLoading}
            >
              Cancel
            </button>
            <button
              type="submit"
              className="flex-1 py-2 px-4 bg-blue-600 text-white rounded-lg font-semibold hover:bg-blue-700 transition disabled:opacity-50"
              disabled={isLoading}
            >
              {isLoading ? "Creating..." : "Create Module"}
            </button>
          </div>
        </form>
      </div>
    </div>
  );
}

// ─── Add Content Item Modal ──────────────────────────────────────────────────

function AddContentItemModal({
  moduleId,
  isOpen,
  onClose,
  onSuccess,
  token,
  existingItemsCount,
}) {
  const [contentType, setContentType] = useState("article");
  const [formData, setFormData] = useState({
    title: "",
    estimated_duration_minutes: 15,
    article: { body: "", summary: "" },
    quiz: {
      total_marks: 10,
      passing_marks: 5,
      questions: [],
    },
  });
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState("");

  const handleInputChange = (e) => {
    const { name, value } = e.target;
    setFormData((prev) => ({
      ...prev,
      [name]: name === "estimated_duration_minutes" ? parseInt(value) : value,
    }));
  };

  const handleContentTypeChange = (type) => {
    setContentType(type);
  };

  const handleAddQuestion = () => {
    setFormData((prev) => ({
      ...prev,
      quiz: {
        ...prev.quiz,
        questions: [
          ...prev.quiz.questions,
          {
            question_text: "",
            options: [
              { text: "", is_correct: true },
              { text: "", is_correct: false },
            ],
          },
        ],
      },
    }));
  };

  const handleUpdateQuestion = (qIdx, updatedQuestion) => {
    setFormData((prev) => ({
      ...prev,
      quiz: {
        ...prev.quiz,
        questions: prev.quiz.questions.map((q, idx) =>
          idx === qIdx ? updatedQuestion : q,
        ),
      },
    }));
  };

  const handleDeleteQuestion = (qIdx) => {
    setFormData((prev) => ({
      ...prev,
      quiz: {
        ...prev.quiz,
        questions: prev.quiz.questions.filter((_, idx) => idx !== qIdx),
      },
    }));
  };

  const handleAddOption = (qIdx) => {
    setFormData((prev) => ({
      ...prev,
      quiz: {
        ...prev.quiz,
        questions: prev.quiz.questions.map((q, idx) =>
          idx === qIdx
            ? {
                ...q,
                options: [...q.options, { text: "", is_correct: false }],
              }
            : q,
        ),
      },
    }));
  };

  const handleUpdateOption = (qIdx, optIdx, field, value) => {
    setFormData((prev) => ({
      ...prev,
      quiz: {
        ...prev.quiz,
        questions: prev.quiz.questions.map((q, idx) =>
          idx === qIdx
            ? {
                ...q,
                options: q.options.map((opt, oIdx) =>
                  oIdx === optIdx ? { ...opt, [field]: value } : opt,
                ),
              }
            : q,
        ),
      },
    }));
  };

  const handleDeleteOption = (qIdx, optIdx) => {
    setFormData((prev) => ({
      ...prev,
      quiz: {
        ...prev.quiz,
        questions: prev.quiz.questions.map((q, idx) =>
          idx === qIdx
            ? {
                ...q,
                options: q.options.filter((_, oIdx) => oIdx !== optIdx),
              }
            : q,
        ),
      },
    }));
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setError("");

    if (!formData.title.trim()) {
      setError("Content title is required");
      return;
    }

    if (contentType === "quiz") {
      if (formData.quiz.questions.length === 0) {
        setError("Quiz must have at least one question");
        return;
      }

      for (const question of formData.quiz.questions) {
        if (!question.question_text.trim()) {
          setError("All questions must have text");
          return;
        }
        if (question.options.length === 0) {
          setError("All questions must have options");
          return;
        }
        if (!question.options.some((opt) => opt.is_correct)) {
          setError("All questions must have at least one correct answer");
          return;
        }
        for (const option of question.options) {
          if (!option.text.trim()) {
            setError("All options must have text");
            return;
          }
        }
      }
    }

    if (contentType === "article" && !formData.article.body.trim()) {
      setError("Article body is required");
      return;
    }

    try {
      setIsLoading(true);

      const contentPayload = {
        module: moduleId,
        title: formData.title,
        content_type: contentType,
        difficulty: "easy",
        estimated_duration_minutes: formData.estimated_duration_minutes,
        order: existingItemsCount + 1,
        is_published: true,
      };

      const contentRes = await fetch(`${API_URL}/api/content-items/`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          Authorization: `Bearer ${token}`,
        },
        body: JSON.stringify(contentPayload),
      });

      if (!contentRes.ok) {
        const errorData = await contentRes.json();
        throw new Error(
          errorData.detail || `Failed to create content: ${contentRes.status}`,
        );
      }

      const createdContent = await contentRes.json();
      const contentItemId = createdContent.id;

      // Create article if article type
      if (contentType === "article") {
        const articlePayload = {
          content_item: contentItemId,
          body: formData.article.body,
          summary: formData.article.summary,
        };

        const articleRes = await fetch(`${API_URL}/api/courses/articles/`, {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
            Authorization: `Bearer ${token}`,
          },
          body: JSON.stringify(articlePayload),
        });

        if (!articleRes.ok) {
          throw new Error(`Failed to create article: ${articleRes.status}`);
        }
      }

      // Create quiz if quiz type
      if (contentType === "quiz") {
        const quizPayload = {
          content_item: contentItemId,
          total_marks: formData.quiz.total_marks,
          passing_marks: formData.quiz.passing_marks,
        };

        const quizRes = await fetch(`${API_URL}/api/courses/quizzes/`, {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
            Authorization: `Bearer ${token}`,
          },
          body: JSON.stringify(quizPayload),
        });

        if (!quizRes.ok) {
          throw new Error(`Failed to create quiz: ${quizRes.status}`);
        }

        const createdQuiz = await quizRes.json();
        const quizId = createdQuiz.id;

        // Create questions and options
        for (let qIdx = 0; qIdx < formData.quiz.questions.length; qIdx++) {
          const question = formData.quiz.questions[qIdx];

          const questionPayload = {
            quiz: quizId,
            question_text: question.question_text,
            order: qIdx + 1,
          };

          const questionRes = await fetch(
            `${API_URL}/api/courses/quiz-questions/`,
            {
              method: "POST",
              headers: {
                "Content-Type": "application/json",
                Authorization: `Bearer ${token}`,
              },
              body: JSON.stringify(questionPayload),
            },
          );

          if (!questionRes.ok) {
            throw new Error(`Failed to create question: ${questionRes.status}`);
          }

          const createdQuestion = await questionRes.json();
          const questionId = createdQuestion.id;

          // Create options
          for (const option of question.options) {
            const optionPayload = {
              question: questionId,
              option_text: option.text,
              is_correct: option.is_correct,
            };

            const optionRes = await fetch(
              `${API_URL}/api/courses/quiz-options/`,
              {
                method: "POST",
                headers: {
                  "Content-Type": "application/json",
                  Authorization: `Bearer ${token}`,
                },
                body: JSON.stringify(optionPayload),
              },
            );

            if (!optionRes.ok) {
              throw new Error(`Failed to create option: ${optionRes.status}`);
            }
          }
        }
      }

      onSuccess();
      onClose();
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to create content");
    } finally {
      setIsLoading(false);
    }
  };

  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50 overflow-y-auto">
      <div className="bg-white rounded-2xl p-6 w-full max-w-2xl my-8">
        <div className="flex items-center justify-between mb-4">
          <h2 className="text-xl font-bold text-gray-900">Add Content Item</h2>
          <button
            onClick={onClose}
            className="p-1 text-gray-400 hover:text-gray-600 transition"
          >
            <X className="w-5 h-5" />
          </button>
        </div>

        {error && (
          <div className="mb-4 p-3 bg-red-50 border border-red-200 rounded-lg text-sm text-red-700">
            {error}
          </div>
        )}

        <form
          onSubmit={handleSubmit}
          className="space-y-4 max-h-96 overflow-y-auto"
        >
          {/* Content Type */}
          <div>
            <label className="block text-sm font-semibold text-gray-900 mb-2">
              Content Type *
            </label>
            <div className="grid grid-cols-2 gap-2">
              {[
                { value: "article", label: "Article", icon: FileText },
                { value: "quiz", label: "Quiz", icon: HelpCircle },
              ].map((type) => (
                <button
                  key={type.value}
                  type="button"
                  onClick={() => handleContentTypeChange(type.value)}
                  className={`p-3 rounded-lg border-2 font-semibold transition-all flex items-center justify-center gap-2 ${
                    contentType === type.value
                      ? "border-blue-600 bg-blue-50 text-blue-600"
                      : "border-gray-200 bg-white text-gray-700 hover:border-blue-300"
                  }`}
                >
                  <type.icon className="w-4 h-4" />
                  {type.label}
                </button>
              ))}
            </div>
          </div>

          {/* Common Fields */}
          <div className="grid grid-cols-2 gap-3">
            <div>
              <label className="block text-sm font-semibold text-gray-900 mb-1">
                Title *
              </label>
              <input
                type="text"
                name="title"
                value={formData.title}
                onChange={handleInputChange}
                className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent outline-none"
                placeholder="e.g., React Basics"
              />
            </div>

            <div>
              <label className="block text-sm font-semibold text-gray-900 mb-1">
                Duration (min) *
              </label>
              <input
                type="number"
                name="estimated_duration_minutes"
                value={formData.estimated_duration_minutes}
                onChange={handleInputChange}
                min="1"
                className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent outline-none"
              />
            </div>
          </div>

          {/* Article Content */}
          {contentType === "article" && (
            <div className="space-y-3">
              <div>
                <label className="block text-sm font-semibold text-gray-900 mb-1">
                  Summary
                </label>
                <input
                  type="text"
                  value={formData.article.summary}
                  onChange={(e) =>
                    setFormData((prev) => ({
                      ...prev,
                      article: { ...prev.article, summary: e.target.value },
                    }))
                  }
                  className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent outline-none"
                  placeholder="Brief summary"
                />
              </div>

              <div>
                <label className="block text-sm font-semibold text-gray-900 mb-1">
                  Body *
                </label>
                <textarea
                  value={formData.article.body}
                  onChange={(e) =>
                    setFormData((prev) => ({
                      ...prev,
                      article: { ...prev.article, body: e.target.value },
                    }))
                  }
                  className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent outline-none"
                  rows="3"
                  placeholder="Article content..."
                />
              </div>
            </div>
          )}

          {/* Quiz Content */}
          {contentType === "quiz" && (
            <div className="space-y-3">
              <div className="grid grid-cols-2 gap-3">
                <div>
                  <label className="block text-sm font-semibold text-gray-900 mb-1">
                    Total Marks *
                  </label>
                  <input
                    type="number"
                    value={formData.quiz.total_marks}
                    onChange={(e) =>
                      setFormData((prev) => ({
                        ...prev,
                        quiz: {
                          ...prev.quiz,
                          total_marks: parseInt(e.target.value) || 10,
                        },
                      }))
                    }
                    min="1"
                    className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent outline-none"
                  />
                </div>

                <div>
                  <label className="block text-sm font-semibold text-gray-900 mb-1">
                    Passing Marks *
                  </label>
                  <input
                    type="number"
                    value={formData.quiz.passing_marks}
                    onChange={(e) =>
                      setFormData((prev) => ({
                        ...prev,
                        quiz: {
                          ...prev.quiz,
                          passing_marks: parseInt(e.target.value) || 5,
                        },
                      }))
                    }
                    min="1"
                    className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent outline-none"
                  />
                </div>
              </div>

              {/* Questions */}
              <div>
                <div className="flex items-center justify-between mb-2">
                  <label className="block text-sm font-semibold text-gray-900">
                    Questions ({formData.quiz.questions.length})
                  </label>
                  <button
                    type="button"
                    onClick={handleAddQuestion}
                    className="text-xs text-blue-600 hover:text-blue-700 font-semibold flex items-center gap-1"
                  >
                    <Plus className="w-3 h-3" />
                    Add Question
                  </button>
                </div>

                <div className="space-y-2">
                  {formData.quiz.questions.map((question, qIdx) => (
                    <QuestionEditor
                      key={qIdx}
                      question={question}
                      questionIndex={qIdx}
                      onUpdate={handleUpdateQuestion}
                      onDelete={handleDeleteQuestion}
                      onAddOption={handleAddOption}
                      onUpdateOption={handleUpdateOption}
                      onDeleteOption={handleDeleteOption}
                    />
                  ))}
                </div>
              </div>
            </div>
          )}

          <div className="flex gap-3 pt-4 border-t border-gray-200">
            <button
              type="button"
              onClick={onClose}
              className="flex-1 py-2 px-4 border border-gray-300 text-gray-900 rounded-lg font-semibold hover:bg-gray-50 transition"
              disabled={isLoading}
            >
              Cancel
            </button>
            <button
              type="submit"
              className="flex-1 py-2 px-4 bg-blue-600 text-white rounded-lg font-semibold hover:bg-blue-700 transition disabled:opacity-50"
              disabled={isLoading}
            >
              {isLoading ? "Creating..." : "Add Content"}
            </button>
          </div>
        </form>
      </div>
    </div>
  );
}

// ─── Question Editor Component ────────────────────────────────────────────────

function QuestionEditor({
  question,
  questionIndex,
  onUpdate,
  onDelete,
  onAddOption,
  onUpdateOption,
  onDeleteOption,
}) {
  const [isExpanded, setIsExpanded] = useState(true);
  const hasCorrectAnswer = question.options?.some((opt) => opt.is_correct);

  return (
    <div className="border border-gray-200 rounded-lg p-3 bg-gray-50">
      <div className="flex items-start justify-between gap-2">
        <button
          type="button"
          onClick={() => setIsExpanded(!isExpanded)}
          className="flex-1 text-left flex items-center justify-between hover:opacity-70"
        >
          <div>
            <h5 className="font-semibold text-gray-900 text-sm">
              Q{questionIndex + 1}: {question.question_text || "Untitled"}
            </h5>
            {!hasCorrectAnswer && (
              <p className="text-xs text-red-600 mt-0.5">
                ⚠️ No correct answer
              </p>
            )}
          </div>
          {isExpanded ? (
            <ChevronUp className="w-4 h-4 text-gray-500" />
          ) : (
            <ChevronDown className="w-4 h-4 text-gray-500" />
          )}
        </button>
        <button
          type="button"
          onClick={() => onDelete(questionIndex)}
          className="p-1 text-red-600 hover:bg-red-50 rounded"
        >
          <Trash2 className="w-4 h-4" />
        </button>
      </div>

      {isExpanded && (
        <div className="mt-3 space-y-3 pt-3 border-t border-gray-200">
          {/* Question Text */}
          <div>
            <label className="text-xs font-semibold text-gray-900 block mb-1">
              Question *
            </label>
            <textarea
              value={question.question_text}
              onChange={(e) =>
                onUpdate(questionIndex, {
                  ...question,
                  question_text: e.target.value,
                })
              }
              className="w-full px-2 py-1.5 border border-gray-300 rounded text-sm focus:ring-2 focus:ring-blue-500 focus:border-transparent outline-none"
              rows="2"
              placeholder="Enter question..."
            />
          </div>

          {/* Options */}
          <div>
            <div className="flex items-center justify-between mb-1.5">
              <label className="text-xs font-semibold text-gray-900">
                Options *
              </label>
              <button
                type="button"
                onClick={() => onAddOption(questionIndex)}
                disabled={question.options?.length >= 6}
                className="text-xs text-blue-600 hover:text-blue-700 font-semibold disabled:opacity-50"
              >
                + Add Option
              </button>
            </div>

            <div className="space-y-1.5">
              {question.options?.map((option, optIdx) => (
                <div key={optIdx} className="flex items-center gap-2">
                  <input
                    type="checkbox"
                    checked={option.is_correct}
                    onChange={(e) =>
                      onUpdateOption(
                        questionIndex,
                        optIdx,
                        "is_correct",
                        e.target.checked,
                      )
                    }
                    className="w-4 h-4 accent-blue-600"
                  />
                  <input
                    type="text"
                    value={option.text}
                    onChange={(e) =>
                      onUpdateOption(
                        questionIndex,
                        optIdx,
                        "text",
                        e.target.value,
                      )
                    }
                    placeholder={`Option ${String.fromCharCode(65 + optIdx)}`}
                    className="flex-1 px-2 py-1.5 border border-gray-300 rounded text-sm focus:ring-2 focus:ring-blue-500 focus:border-transparent outline-none"
                  />
                  {question.options.length > 2 && (
                    <button
                      type="button"
                      onClick={() => onDeleteOption(questionIndex, optIdx)}
                      className="p-1 text-red-600 hover:bg-red-50 rounded"
                    >
                      <X className="w-4 h-4" />
                    </button>
                  )}
                </div>
              ))}
            </div>
            <p className="text-xs text-gray-500 mt-1.5">
              ✓ Check the correct answer(s)
            </p>
          </div>
        </div>
      )}
    </div>
  );
}

// ─── Article Preview ──────────────────────────────────────────────────────────

function ArticlePreview({ item }) {
  const [expanded, setExpanded] = useState(false);

  return (
    <div className="mt-3">
      <button
        onClick={() => setExpanded((v) => !v)}
        className="flex items-center gap-2 text-xs font-semibold text-blue-600 bg-blue-50 hover:bg-blue-100 border border-blue-200 px-3 py-1.5 rounded-lg transition-all"
      >
        <FileText className="w-3.5 h-3.5" />
        {expanded ? "Hide Preview" : "Preview Content"}
        {expanded ? (
          <ChevronUp className="w-3 h-3" />
        ) : (
          <ChevronDown className="w-3 h-3" />
        )}
      </button>

      {expanded && (
        <div className="mt-3 border border-gray-200 rounded-xl bg-gray-50 p-4 text-sm text-gray-700 leading-relaxed">
          {item.content_data?.body ? (
            <div
              className="prose prose-sm max-w-none"
              dangerouslySetInnerHTML={{ __html: item.content_data.body }}
            />
          ) : (
            <div className="space-y-3 text-gray-500">
              <p className="font-semibold text-gray-900">{item.title}</p>
              <p>
                Content preview not available. Full content will display when
                published.
              </p>
              <div className="flex items-center gap-2 pt-2 border-t border-gray-200 text-xs text-gray-400">
                <Clock className="w-3.5 h-3.5" />~
                {item.estimated_duration_minutes} min read
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
}

// ─── Content Item Card ────────────────────────────────────────────────────────

function ContentItemCard({ item }) {
  const config =
    contentTypeConfig[item.content_type] || contentTypeConfig.article;
  const Icon = config.icon;
  function handleDelete() {
    if (item.content_type === "article") {
      // call delete article api
    } else if (item.content_type === "quiz") {
      // call delete quiz api
    }
  }

  return (
    <div className="border border-gray-100 rounded-xl p-4 hover:border-blue-200 hover:shadow-sm transition-all bg-white group">
      <div className="flex items-start justify-between gap-3 mb-3">
        <div className="flex items-center gap-3 flex-1 min-w-0">
          <div
            className={`w-8 h-8 rounded-lg flex items-center justify-center flex-shrink-0 ${config.badge} border`}
          >
            <Icon className="w-4 h-4" />
          </div>
          <div className="flex-1 min-w-0">
            <p className="text-sm font-semibold text-gray-900 truncate">
              {item.title}
            </p>
            <div className="flex items-center gap-2 mt-0.5 flex-wrap">
              <span
                className={`text-xs px-2 py-0.5 rounded-full border font-medium ${config.badge}`}
              >
                {config.label}
              </span>
              <span className="text-xs text-gray-400 flex items-center gap-1">
                <Clock className="w-3 h-3" />
                {item.estimated_duration_minutes}m
              </span>
            </div>
          </div>
        </div>
        <div className="flex items-center gap-2 flex-shrink-0 opacity-0 group-hover:opacity-100 transition-opacity">
          <button
            onClick={handleDelete}
            className="p-1.5 text-gray-400 hover:text-red-600 hover:bg-red-50 rounded-lg transition"
          >
            <Trash2 className="w-4 h-4" />
          </button>
        </div>
      </div>

      {item.content_type === "article" && <ArticlePreview item={item} />}
      {item.content_type === "quiz" && <QuizPreview item={item} />}
    </div>
  );
}
function QuizPreview({ item }) {
  const totalQuestions =
    item.content_data?.questions?.length || item.total_questions || 0;
  return (
    <div className="mt-3 text-xs text-gray-600 bg-gray-50 rounded-lg p-3 border">
      <p className="font-medium text-gray-800 mb-1">Quiz Preview</p>

      <div className="flex items-center gap-4 text-gray-500">
        <span>📝 {totalQuestions} Questions</span>
        <span>⏱ {item.estimated_duration_minutes || 0} mins</span>
      </div>
    </div>
  );
}
// ─── Module Card ──────────────────────────────────────────────────────────────

function ModuleCard({ moduleData, index, onAddItem, token }) {
  const [open, setOpen] = useState(false);
  const [showAddModal, setShowAddModal] = useState(false);

  const { data, isLoading, error, refetch } = useQuery({
    queryKey: ["modules", moduleData.id],
    queryFn: () => getCourseModules(moduleData.id),
    enabled: open,
  });

  function handleOpen() {
    setOpen((v) => !v);
    refetch();
  }

  const module = data;
  const totalMinutes =
    module?.content_items?.reduce(
      (sum, item) => sum + (item.estimated_duration_minutes || 0),
      0,
    ) || 0;

  // Determine which data to show in header
  const displayTitle = module?.title || moduleData.title;
  const displayDescription = module?.description || moduleData.description;
  const displayDuration = module
    ? totalMinutes
    : moduleData.estimated_duration || "";
  const displayItemCount =
    module?.content_items?.length || moduleData.content_items?.length || "";

  return (
    <>
      <div className="bg-white border border-gray-100 rounded-xl shadow-sm hover:shadow-md transition-all overflow-hidden">
        {/* Module Header */}
        <button
          onClick={handleOpen}
          className="w-full flex items-center justify-between gap-4 p-5 text-left hover:bg-gray-50 transition-colors"
        >
          <div className="flex items-center gap-4 flex-1 min-w-0">
            <div className="w-10 h-10 bg-gradient-to-br from-blue-500 to-purple-600 rounded-xl flex items-center justify-center text-white font-bold text-sm flex-shrink-0">
              {String(index + 1).padStart(2, "0")}
            </div>
            <div className="flex-1 min-w-0">
              <p className="font-bold text-gray-900 text-sm">{displayTitle}</p>
              <p className="text-xs text-gray-500 line-clamp-1 mt-0.5">
                {displayDescription}
              </p>
            </div>
          </div>
          <div className="flex items-center gap-4 flex-shrink-0">
            <div className="hidden sm:flex items-center gap-3 text-xs text-gray-400">
              {!isLoading && (
                <>
                  {displayItemCount && (
                    <span className="flex items-center gap-1">
                      <Layers className="w-3.5 h-3.5" />
                      {displayItemCount} items
                    </span>
                  )}
                  {displayDuration && (
                    <span className="flex items-center gap-1">
                      <Clock className="w-3.5 h-3.5" />
                      {displayDuration}m
                    </span>
                  )}
                </>
              )}
              {isLoading && (
                <span className="text-xs text-gray-400">Loading...</span>
              )}
            </div>
            {/* <button className="p-1.5 text-gray-400 hover:text-blue-600 hover:bg-blue-50 rounded-lg transition">
              <Edit className="w-4 h-4" />
            </button> */}
            {open ? (
              <ChevronUp className="w-4 h-4 text-gray-400" />
            ) : (
              <ChevronDown className="w-4 h-4 text-gray-400" />
            )}
          </div>
        </button>

        {/* Module Body */}
        {open && isLoading && (
          <div className="border-t border-gray-100 bg-gray-50 p-4">
            <div className="w-full animate-pulse space-y-3">
              <div className="h-4 bg-gray-300 rounded w-1/3"></div>
              <div className="h-3 bg-gray-300 rounded w-full"></div>
              <div className="h-3 bg-gray-300 rounded w-5/6"></div>
            </div>
          </div>
        )}

        {open && error && (
          <div className="border-t border-red-200 bg-red-50 p-4">
            <p className="text-sm text-red-700">
              Error loading module details.
            </p>
          </div>
        )}

        {open && module && (
          <div className="border-t border-gray-100 p-5 space-y-4">
            {/* Learning Objectives */}
            {module.learning_objectives?.length > 0 && (
              <div>
                <p className="text-xs font-semibold text-gray-500 uppercase tracking-wider mb-2">
                  Learning Objectives
                </p>
                <ul className="space-y-1.5">
                  {module.learning_objectives.slice(0, 3).map((obj, i) => (
                    <li
                      key={i}
                      className="flex items-start gap-2 text-xs text-gray-600"
                    >
                      <CheckCircle className="w-3.5 h-3.5 text-green-500 flex-shrink-0 mt-0.5" />
                      {typeof obj === "string"
                        ? obj
                        : (obj.title ?? JSON.stringify(obj))}
                    </li>
                  ))}
                  {module.learning_objectives.length > 3 && (
                    <li className="text-xs text-blue-600 font-medium ml-5">
                      +{module.learning_objectives.length - 3} more objectives
                    </li>
                  )}
                </ul>
              </div>
            )}

            {/* Content Items */}
            {module.content_items?.length > 0 && (
              <div>
                <div className="flex items-center justify-between mb-2">
                  <p className="text-xs font-semibold text-gray-500 uppercase tracking-wider">
                    Content ({module.content_items.length})
                  </p>
                  <button
                    onClick={() => setShowAddModal(true)}
                    className="text-xs text-blue-600 hover:text-blue-700 font-semibold transition"
                  >
                    + Add Item
                  </button>
                </div>
                <div className="space-y-2">
                  {module.content_items.map((item) => (
                    <ContentItemCard key={item.id} item={item} />
                  ))}
                </div>
              </div>
            )}

            {/* Add Item Button when no items */}
            {module.content_items?.length === 0 && (
              <div>
                <p className="text-xs font-semibold text-gray-500 uppercase tracking-wider mb-2">
                  Content
                </p>
                <button
                  onClick={() => setShowAddModal(true)}
                  className="w-full py-2.5 px-4 border border-blue-200 text-blue-600 rounded-lg font-semibold text-sm hover:bg-blue-50 transition"
                >
                  + Add First Item
                </button>
              </div>
            )}
          </div>
        )}
      </div>

      <AddContentItemModal
        moduleId={module?.id || moduleData.id}
        isOpen={showAddModal}
        onClose={() => setShowAddModal(false)}
        onSuccess={() => {
          setShowAddModal(false);
          onAddItem();
        }}
        token={token}
        existingItemsCount={
          module?.content_items?.length || moduleData.content_items?.length || 0
        }
      />
    </>
  );
}

// ─── Main Page ────────────────────────────────────────────────────────────────

export default function InstructorCourseDetail() {
  const { courseId } = useParams();
  const navigate = useNavigate();
  const id = courseId ? parseInt(courseId) : 0;

  const { data: course, isLoading, error, refetch } = useCourseDetail(id);
  const { logout, user } = useAuth();

  const [showAddModuleModal, setShowAddModuleModal] = useState(false);

  const token = localStorage.getItem("access_token");

  const level = course?.difficulty_level?.toLowerCase();
  const gradient = levelGradients[level] || "from-blue-500 to-indigo-600";
  const icon = levelIcons[level] || "📖";

  const initials = user
    ? `${user.firstName?.[0] ?? ""}${user.lastName?.[0] ?? ""}`.toUpperCase()
    : "I";

  const handleLogout = () => {
    logout();
    navigate("/login");
  };

  const handleRefresh = () => {
    refetch();
  };

  // ── Loading ──────────────────────────────────────────────────────────────

  if (isLoading) {
    return <InstructorCourseDetailSkeleton />;
  }

  // ── Error ────────────────────────────────────────────────────────────────

  if (error || !course) {
    return (
      <div className="min-h-screen bg-gray-50 flex items-center justify-center">
        <div className="text-center">
          <p className="text-xl text-gray-500 mb-4">
            {error instanceof Error ? error.message : "Course not found."}
          </p>
          <Link
            to="/instructor"
            className="px-5 py-2 bg-blue-600 text-white rounded-xl font-semibold text-sm hover:bg-blue-700 transition"
          >
            Back to Dashboard
          </Link>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gray-50">
      {/* ── Navbar ── */}
      <nav className="bg-white shadow-sm border-b sticky top-0 z-50">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex justify-between items-center h-16">
            <Link to="/" className="flex items-center gap-2">
              <div className="bg-gradient-to-r from-blue-600 to-purple-600 p-2 rounded-lg">
                <svg
                  xmlns="http://www.w3.org/2000/svg"
                  width="20"
                  height="20"
                  fill="none"
                  stroke="white"
                  strokeWidth="2"
                >
                  <polyline points="16 18 22 12 16 6" />
                  <polyline points="8 6 2 12 8 18" />
                </svg>
              </div>
              <span className="text-xl font-bold bg-gradient-to-r from-blue-600 to-purple-600 bg-clip-text text-transparent">
                Vision-Code
              </span>
            </Link>
            <div className="flex items-center gap-3">
              <Link
                to="/settings"
                className="p-2 text-gray-500 hover:text-blue-600 transition rounded-lg hover:bg-gray-100"
              >
                <Settings className="w-5 h-5" />
              </Link>
              <div className="w-9 h-9 bg-gradient-to-r from-blue-600 to-purple-600 rounded-full flex items-center justify-center text-white font-bold text-sm">
                {initials}
              </div>
              <button
                onClick={handleLogout}
                className="flex items-center gap-1.5 px-3 py-2 text-sm text-red-600 hover:bg-red-50 rounded-lg transition font-medium"
              >
                <LogOut className="w-4 h-4" />
                Logout
              </button>
            </div>
          </div>
        </div>
      </nav>

      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {/* ── Back Link ── */}
        <Link
          to="/instructor"
          className="inline-flex items-center gap-2 text-gray-500 hover:text-blue-600 text-sm font-medium mb-6 transition-colors"
        >
          <ArrowLeft className="w-4 h-4" /> Back to Dashboard
        </Link>

        {/* ── Hero Banner ── */}
        <div
          className={`bg-gradient-to-r ${gradient} rounded-2xl p-8 mb-8 text-white relative overflow-hidden`}
        >
          <div className="absolute top-0 right-0 w-64 h-64 bg-white/5 rounded-full -translate-y-1/2 translate-x-1/2" />
          <div className="absolute bottom-0 left-1/2 w-48 h-48 bg-white/5 rounded-full translate-y-1/2" />
          <div className="relative z-10 flex items-start justify-between gap-6 flex-wrap">
            <div className="flex-1">
              <div className="flex items-center gap-4 mb-4">
                <div className="w-16 h-16 bg-white/20 rounded-2xl flex items-center justify-center text-3xl flex-shrink-0 shadow-lg backdrop-blur-sm">
                  {icon}
                </div>
                <div className="flex flex-wrap items-center gap-2">
                  <span className="text-xs font-semibold bg-white/20 px-2 py-0.5 rounded-full capitalize">
                    {course.difficulty_level}
                  </span>
                  <span className="text-xs font-semibold bg-blue-400/30 border border-blue-300/50 px-2 py-0.5 rounded-full">
                    Your Course
                  </span>
                </div>
              </div>
              <h1 className="text-2xl sm:text-3xl font-bold mb-2">
                {course.title}
              </h1>
              <p className="text-white/80 text-sm max-w-2xl">
                {course.description}
              </p>
              <div className="flex items-center gap-4 mt-3 text-sm text-white/70 flex-wrap">
                <span className="flex items-center gap-1.5">
                  <Users className="w-4 h-4" />
                  {course.total_students?.toLocaleString()} students
                </span>
                <span className="flex items-center gap-1.5">
                  <Layers className="w-4 h-4" />
                  {course.total_modules} modules
                </span>
                <span className="flex items-center gap-1.5">
                  <Star className="w-4 h-4 fill-white/70" />
                  4.5 rating
                </span>
              </div>
            </div>
            {/* <button className="px-4 py-2.5 bg-white text-blue-600 rounded-xl font-bold text-sm hover:bg-blue-50 transition flex items-center gap-2 flex-shrink-0 h-fit">
              <Edit className="w-4 h-4" />
              Edit Course
            </button> */}
          </div>
        </div>

        {/* ── Course Stats ── */}
        <div className="grid grid-cols-2 sm:grid-cols-4 gap-4 mb-8">
          {[
            {
              label: "Total Students",
              value: course.total_students?.toLocaleString(),
              icon: Users,
            },
            { label: "Modules", value: course.total_modules, icon: Layers },
            { label: "Rating", value: "4.5 / 5", icon: Star },
            {
              label: "Level",
              value:
                course.difficulty_level?.charAt(0).toUpperCase() +
                course.difficulty_level?.slice(1),
              icon: BarChart2,
            },
          ].map(({ label, value, icon: Icon }) => (
            <div
              key={label}
              className="bg-white rounded-xl shadow-sm border border-gray-100 p-4"
            >
              <div className="flex items-center gap-2 mb-2">
                <Icon className="w-4 h-4 text-blue-500" />
                <p className="text-xs text-gray-500 font-medium">{label}</p>
              </div>
              <p className="text-lg font-bold text-gray-900">{value}</p>
            </div>
          ))}
        </div>

        {/* ── Main Content ── */}
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
          {/* Left — Modules */}
          <div className="lg:col-span-2 space-y-6">
            {/* Instructor Info */}
            <div className="bg-white rounded-xl shadow-sm border border-gray-100 p-5 flex items-center gap-4">
              <div className="w-12 h-12 bg-gradient-to-br from-blue-500 to-purple-600 rounded-xl flex items-center justify-center text-white font-bold text-lg flex-shrink-0">
                {course.instructor.name?.charAt(0).toUpperCase()}
              </div>
              <div>
                <p className="text-xs text-gray-400 font-medium">
                  Course Instructor
                </p>
                <p className="font-bold text-gray-900">
                  {course.instructor.name}
                </p>
                <p className="text-xs text-gray-500">
                  {course.instructor.email}
                </p>
              </div>
            </div>

            {/* Modules */}
            <div>
              <div className="flex items-center justify-between mb-4">
                <h2 className="text-xl font-bold text-gray-900">
                  Course Modules
                </h2>
                <button
                  onClick={() => setShowAddModuleModal(true)}
                  className="text-sm text-blue-600 hover:text-blue-700 font-semibold transition"
                >
                  + New Module
                </button>
              </div>
              <div className="space-y-3">
                {course.modules.map((module, idx) => (
                  <ModuleCard
                    key={module.id}
                    moduleData={module}
                    index={idx}
                    onAddItem={handleRefresh}
                    token={token}
                  />
                ))}
              </div>
            </div>
          </div>

          {/* Right — Sidebar */}
          <div className="space-y-5">
            {/* Course Actions */}
            <div className="bg-white rounded-xl shadow-sm border border-gray-100 p-5">
              <h3 className="font-bold text-gray-900 mb-4">Course Actions</h3>
              <div className="space-y-2">
                {/* <button className="w-full py-2.5 px-4 bg-blue-600 text-white rounded-lg font-semibold text-sm hover:bg-blue-700 transition flex items-center justify-center gap-2">
                  <Edit className="w-4 h-4" />
                  Edit Course
                </button> */}
                <button className="w-full py-2.5 px-4 bg-gray-100 text-gray-900 rounded-lg font-semibold text-sm hover:bg-gray-200 transition">
                  Preview as Student
                </button>
                <button className="w-full py-2.5 px-4 bg-gray-100 text-gray-900 rounded-lg font-semibold text-sm hover:bg-gray-200 transition">
                  View Analytics
                </button>
              </div>
            </div>

            {/* Quick Stats */}
            <div className="bg-white rounded-xl shadow-sm border border-gray-100 p-5">
              <h3 className="font-bold text-gray-900 mb-4">Engagement</h3>
              <div className="space-y-3">
                {[
                  {
                    label: "Avg. Completion",
                    value: "72%",
                    color: "text-green-600",
                  },
                  {
                    label: "Avg. Rating",
                    value: "4.5/5",
                    color: "text-yellow-600",
                  },
                  {
                    label: "Total Reviews",
                    value: "1,234",
                    color: "text-blue-600",
                  },
                ].map(({ label, value, color }) => (
                  <div
                    key={label}
                    className="flex items-center justify-between"
                  >
                    <p className="text-sm text-gray-600">{label}</p>
                    <span className={`font-bold ${color}`}>{value}</span>
                  </div>
                ))}
              </div>
            </div>

            {/* Topics */}
            {course.topics?.length > 0 && (
              <div className="bg-white rounded-xl shadow-sm border border-gray-100 p-5">
                <h3 className="font-bold text-gray-900 mb-3 flex items-center gap-2">
                  <Tag className="w-4 h-4 text-blue-500" />
                  Topics
                </h3>
                <div className="flex flex-wrap gap-2">
                  {course.topics.slice(0, 8).map((topic) => (
                    <span
                      key={topic}
                      className="text-xs px-2.5 py-1 bg-blue-50 text-blue-700 border border-blue-100 rounded-full font-medium"
                    >
                      {topic}
                    </span>
                  ))}
                  {course.topics.length > 8 && (
                    <span className="text-xs px-2.5 py-1 text-gray-400 font-medium">
                      +{course.topics.length - 8} more
                    </span>
                  )}
                </div>
              </div>
            )}
          </div>
        </div>
      </div>

      {/* Modals */}
      {token && (
        <>
          <AddModuleModal
            courseId={id}
            isOpen={showAddModuleModal}
            onClose={() => setShowAddModuleModal(false)}
            onSuccess={handleRefresh}
            token={token}
          />
        </>
      )}
    </div>
  );
}
