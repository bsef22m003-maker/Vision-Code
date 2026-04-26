import { useState } from "react";
import { useNavigate } from "react-router-dom";
import {
  ArrowLeft,
  ChevronRight,
  ChevronLeft,
  Upload,
  AlertCircle,
  CheckCircle,
  BookOpen,
  Plus,
  Trash2,
  Save,
  X,
  Eye,
  FileText,
  HelpCircle,
  ChevronDown,
  ChevronUp,
  Target,
} from "lucide-react";
import { API_URL } from "../constants";

// ─── Step Indicator ───────────────────────────────────────────────────────────

function StepIndicator({ currentStep, totalSteps }) {
  return (
    <div className="flex items-center justify-between mb-8">
      {Array.from({ length: totalSteps }).map((_, index) => (
        <div key={index} className="flex items-center flex-1">
          <div
            className={`w-10 h-10 rounded-full flex items-center justify-center font-bold text-sm transition-all ${
              index < currentStep
                ? "bg-green-500 text-white"
                : index === currentStep
                  ? "bg-blue-600 text-white ring-4 ring-blue-200"
                  : "bg-gray-200 text-gray-600"
            }`}
          >
            {index < currentStep ? (
              <CheckCircle className="w-6 h-6" />
            ) : (
              index + 1
            )}
          </div>
          {index < totalSteps - 1 && (
            <div
              className={`flex-1 h-1 mx-2 rounded-full transition-all ${
                index < currentStep ? "bg-green-500" : "bg-gray-200"
              }`}
            />
          )}
        </div>
      ))}
    </div>
  );
}

// ─── Step 1: Basic Course Information ──────────────────────────────────────────

function Step1BasicInfo({ data, onChange }) {
  return (
    <div className="space-y-6">
      <div>
        <label className="block text-sm font-semibold text-gray-900 mb-2">
          Course Title *
        </label>
        <input
          type="text"
          name="title"
          value={data.title}
          onChange={onChange}
          placeholder="e.g., DSA for System Design Interviews"
          className="w-full px-4 py-3 border border-gray-200 rounded-xl focus:outline-none focus:border-blue-500 focus:ring-2 focus:ring-blue-50"
        />
      </div>

      <div>
        <label className="block text-sm font-semibold text-gray-900 mb-2">
          Course Description *
        </label>
        <textarea
          name="description"
          value={data.description}
          onChange={onChange}
          placeholder="Describe your course, learning outcomes, and target audience..."
          rows="5"
          className="w-full px-4 py-3 border border-gray-200 rounded-xl focus:outline-none focus:border-blue-500 focus:ring-2 focus:ring-blue-50"
        />
      </div>

      <div className="grid grid-cols-2 gap-4">
        <div>
          <label className="block text-sm font-semibold text-gray-900 mb-2">
            Difficulty Level *
          </label>
          <select
            name="difficulty_level"
            value={data.difficulty_level}
            onChange={onChange}
            className="w-full px-4 py-3 border border-gray-200 rounded-xl focus:outline-none focus:border-blue-500 focus:ring-2 focus:ring-blue-50"
          >
            <option value="beginner">Beginner</option>
            <option value="intermediate">Intermediate</option>
            <option value="advanced">Advanced</option>
          </select>
        </div>

        <div>
          <label className="block text-sm font-semibold text-gray-900 mb-2">
            Price (USD) *
          </label>
          <div className="relative">
            <span className="absolute left-4 top-3 text-gray-600 font-semibold">
              $
            </span>
            <input
              type="number"
              name="price"
              value={data.price}
              onChange={onChange}
              placeholder="1799.00"
              min="0"
              step="0.01"
              className="w-full pl-8 pr-4 py-3 border border-gray-200 rounded-xl focus:outline-none focus:border-blue-500 focus:ring-2 focus:ring-blue-50"
            />
          </div>
        </div>
      </div>

      <div>
        <label className="block text-sm font-semibold text-gray-900 mb-2">
          Course Type *
        </label>
        <div className="grid grid-cols-2 gap-4">
          {[
            { value: "free", label: "Free" },
            { value: "instructor_published", label: "Instructor Published" },
          ].map((option) => (
            <button
              key={option.value}
              onClick={() =>
                onChange({
                  target: { name: "course_type", value: option.value },
                })
              }
              className={`p-3 rounded-xl border-2 font-semibold transition-all ${
                data.course_type === option.value
                  ? "border-blue-600 bg-blue-50 text-blue-600"
                  : "border-gray-200 bg-white text-gray-700 hover:border-blue-300"
              }`}
            >
              {option.label}
            </button>
          ))}
        </div>
      </div>
    </div>
  );
}

// ─── Step 2: Course Image & Topics ────────────────────────────────────────────

function Step2ImageAndTopics({ data, onChange, onImageUpload }) {
  const [previewUrl, setPreviewUrl] = useState(null);
  const [isDragging, setIsDragging] = useState(false);

  const handleImageChange = (e) => {
    const file = e.target.files?.[0];
    if (file) processImage(file);
  };

  const processImage = (file) => {
    const reader = new FileReader();
    reader.onloadend = () => {
      setPreviewUrl(reader.result);
      onImageUpload(file);
    };
    reader.readAsDataURL(file);
  };

  const handleDragOver = (e) => {
    e.preventDefault();
    setIsDragging(true);
  };

  const handleDrop = (e) => {
    e.preventDefault();
    setIsDragging(false);
    const file = e.dataTransfer.files?.[0];
    if (file && file.type.startsWith("image/")) {
      processImage(file);
    }
  };

  return (
    <div className="space-y-6">
      {/* Course Image */}
      <div>
        <label className="block text-sm font-semibold text-gray-900 mb-2">
          Course Image
        </label>
        <div
          onDragOver={handleDragOver}
          onDragLeave={() => setIsDragging(false)}
          onDrop={handleDrop}
          className={`border-2 border-dashed rounded-xl p-8 text-center transition-all cursor-pointer ${
            isDragging
              ? "border-blue-500 bg-blue-50"
              : "border-gray-300 hover:border-blue-400"
          }`}
        >
          {previewUrl ? (
            <div className="relative">
              <img
                src={previewUrl}
                alt="Preview"
                className="max-h-64 mx-auto rounded-lg"
              />
              <button
                onClick={() => {
                  setPreviewUrl(null);
                  onImageUpload(null);
                }}
                className="absolute top-2 right-2 p-1 bg-red-500 text-white rounded-full hover:bg-red-600"
              >
                <X className="w-4 h-4" />
              </button>
            </div>
          ) : (
            <div>
              <Upload className="w-12 h-12 text-gray-400 mx-auto mb-3" />
              <p className="font-semibold text-gray-900 mb-1">
                Drag and drop your image
              </p>
              <p className="text-sm text-gray-500 mb-4">
                or click to browse (JPG, PNG - max 5MB)
              </p>
              <input
                type="file"
                accept="image/*"
                onChange={handleImageChange}
                className="hidden"
                id="image-upload"
              />
              <label
                htmlFor="image-upload"
                className="inline-block px-4 py-2 bg-blue-600 text-white rounded-lg font-semibold text-sm hover:bg-blue-700 cursor-pointer"
              >
                Choose Image
              </label>
            </div>
          )}
        </div>
      </div>

      {/* Topics */}
      <div>
        <label className="block text-sm font-semibold text-gray-900 mb-3">
          Course Topics *
        </label>
        <p className="text-xs text-gray-500 mb-3">
          Add topics that will be covered in your course
        </p>
        <div className="space-y-2">
          {data.topics.map((topic, index) => (
            <div key={index} className="flex gap-2">
              <input
                type="text"
                value={topic}
                onChange={(e) => {
                  const newTopics = [...data.topics];
                  newTopics[index] = e.target.value;
                  onChange({
                    target: { name: "topics", value: newTopics },
                  });
                }}
                placeholder={`Topic ${index + 1}`}
                className="flex-1 px-4 py-2 border border-gray-200 rounded-lg focus:outline-none focus:border-blue-500 focus:ring-2 focus:ring-blue-50"
              />
              {data.topics.length > 1 && (
                <button
                  onClick={() => {
                    const newTopics = data.topics.filter((_, i) => i !== index);
                    onChange({
                      target: { name: "topics", value: newTopics },
                    });
                  }}
                  className="px-3 py-2 text-red-600 hover:bg-red-50 rounded-lg transition"
                >
                  <Trash2 className="w-4 h-4" />
                </button>
              )}
            </div>
          ))}
          <button
            onClick={() => {
              onChange({
                target: { name: "topics", value: [...data.topics, ""] },
              });
            }}
            className="w-full px-4 py-2 border-2 border-dashed border-gray-300 text-gray-600 rounded-lg hover:border-blue-400 hover:text-blue-600 transition flex items-center justify-center gap-2"
          >
            <Plus className="w-4 h-4" />
            Add Topic
          </button>
        </div>
      </div>
    </div>
  );
}

// ─── Quiz Question & Options Editor ───────────��───────────────────────────────

function QuizQuestionEditor({
  question,
  onUpdateQuestion,
  onDeleteQuestion,
  questionIndex,
}) {
  const [isExpanded, setIsExpanded] = useState(false);

  const handleAddOption = () => {
    const newQuestion = {
      ...question,
      options: [...question.options, { text: "", is_correct: false }],
    };
    onUpdateQuestion(questionIndex, newQuestion);
  };

  const handleDeleteOption = (optionIndex) => {
    const newQuestion = {
      ...question,
      options: question.options.filter((_, i) => i !== optionIndex),
    };
    onUpdateQuestion(questionIndex, newQuestion);
  };

  const handleUpdateOption = (optionIndex, field, value) => {
    const newQuestion = {
      ...question,
      options: question.options.map((opt, i) =>
        i === optionIndex ? { ...opt, [field]: value } : opt,
      ),
    };
    onUpdateQuestion(questionIndex, newQuestion);
  };

  return (
    <div className="border border-gray-200 rounded-lg p-4 bg-gray-50">
      <div className="flex items-start justify-between gap-3">
        <div className="flex-1">
          <button
            onClick={() => setIsExpanded(!isExpanded)}
            className="w-full flex items-center justify-between hover:opacity-70 transition"
          >
            <h4 className="font-semibold text-gray-900 text-left line-clamp-2">
              Q{questionIndex + 1}.{" "}
              {question.question_text || "Untitled Question"}
            </h4>
            {isExpanded ? (
              <ChevronUp className="w-4 h-4 text-gray-500 flex-shrink-0" />
            ) : (
              <ChevronDown className="w-4 h-4 text-gray-500 flex-shrink-0" />
            )}
          </button>
        </div>
        <button
          onClick={() => onDeleteQuestion(questionIndex)}
          className="p-2 text-red-600 hover:bg-red-50 rounded-lg transition flex-shrink-0"
        >
          <Trash2 className="w-4 h-4" />
        </button>
      </div>

      {isExpanded && (
        <div className="mt-4 space-y-4 pt-4 border-t border-gray-200">
          {/* Question Text */}
          <div>
            <label className="block text-sm font-semibold text-gray-900 mb-2">
              Question *
            </label>
            <textarea
              value={question.question_text}
              onChange={(e) => {
                onUpdateQuestion(questionIndex, {
                  ...question,
                  question_text: e.target.value,
                });
              }}
              placeholder="What is the question?"
              rows="2"
              className="w-full px-3 py-2 border border-gray-200 rounded-lg focus:outline-none focus:border-blue-500"
            />
          </div>

          {/* Options */}
          <div>
            <label className="block text-sm font-semibold text-gray-900 mb-2">
              Options *
            </label>
            <div className="space-y-2">
              {question.options.map((option, optIdx) => (
                <div key={optIdx} className="flex items-center gap-2">
                  <input
                    type="checkbox"
                    checked={option.is_correct}
                    onChange={(e) =>
                      handleUpdateOption(optIdx, "is_correct", e.target.checked)
                    }
                    className="w-4 h-4 rounded border-gray-300"
                  />
                  <input
                    type="text"
                    value={option.text}
                    onChange={(e) =>
                      handleUpdateOption(optIdx, "text", e.target.value)
                    }
                    placeholder={`Option ${String.fromCharCode(65 + optIdx)}`}
                    className="flex-1 px-3 py-2 border border-gray-200 rounded-lg focus:outline-none focus:border-blue-500"
                  />
                  {question.options.length > 2 && (
                    <button
                      onClick={() => handleDeleteOption(optIdx)}
                      className="p-2 text-red-600 hover:bg-red-50 rounded-lg"
                    >
                      <X className="w-4 h-4" />
                    </button>
                  )}
                </div>
              ))}
              {question.options.length < 6 && (
                <button
                  onClick={handleAddOption}
                  className="w-full px-3 py-2 border border-dashed border-gray-300 text-gray-600 rounded-lg hover:border-blue-400 hover:text-blue-600 transition text-sm flex items-center justify-center gap-1"
                >
                  <Plus className="w-3 h-3" />
                  Add Option
                </button>
              )}
            </div>
            <p className="text-xs text-gray-500 mt-2">
              ✓ Check the correct option(s)
            </p>
          </div>
        </div>
      )}
    </div>
  );
}

// ─── Content Item Section ──────────────────────────────────────────────────────

function ContentItemSection({
  contentItem,
  index,
  onUpdate,
  onDelete,
  contentIndex,
}) {
  const [contentType, setContentType] = useState(contentItem.content_type);

  const handleContentTypeChange = (newType) => {
    setContentType(newType);
    if (newType === "article") {
      onUpdate(index, {
        ...contentItem,
        content_type: "article",
        article: { title: "", body: "", summary: "", duration: 0 },
        quiz: null,
      });
    } else {
      onUpdate(index, {
        ...contentItem,
        content_type: "quiz",
        quiz: { total_marks: 10, passing_marks: 5, questions: [] },
        article: null,
      });
    }
  };

  return (
    <div className="border-l-4 border-purple-400 bg-purple-50 rounded-lg p-4 space-y-4">
      <div className="flex items-center justify-between">
        <h4 className="font-semibold text-gray-900">
          Content Item {contentIndex + 1}
        </h4>
        <button
          onClick={() => onDelete(index)}
          className="p-1 text-red-600 hover:bg-red-100 rounded-lg transition"
        >
          <Trash2 className="w-4 h-4" />
        </button>
      </div>

      {/* Content Type Selection */}
      <div>
        <label className="block text-sm font-semibold text-gray-900 mb-2">
          Content Type *
        </label>
        <div className="grid grid-cols-2 gap-2">
          {[
            { value: "article", icon: FileText, label: "Article" },
            { value: "quiz", icon: HelpCircle, label: "Quiz" },
          ].map((type) => (
            <button
              key={type.value}
              onClick={() => handleContentTypeChange(type.value)}
              className={`p-3 rounded-lg border-2 font-semibold transition-all flex items-center justify-center gap-2 ${
                contentType === type.value
                  ? "border-purple-600 bg-white text-purple-600"
                  : "border-gray-200 bg-white text-gray-700 hover:border-purple-300"
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
          <label className="block text-sm font-semibold text-gray-900 mb-2">
            Title *
          </label>
          <input
            type="text"
            value={contentItem.title}
            onChange={(e) =>
              onUpdate(index, { ...contentItem, title: e.target.value })
            }
            placeholder="e.g., Intro to Django"
            className="w-full px-3 py-2 border border-gray-200 rounded-lg focus:outline-none focus:border-purple-500"
          />
        </div>

        <div>
          <label className="block text-sm font-semibold text-gray-900 mb-2">
            Duration (min)
          </label>
          <input
            type="number"
            value={contentItem.estimated_duration_minutes}
            onChange={(e) =>
              onUpdate(index, {
                ...contentItem,
                estimated_duration_minutes: parseInt(e.target.value) || 0,
              })
            }
            placeholder="10"
            min="1"
            className="w-full px-3 py-2 border border-gray-200 rounded-lg focus:outline-none focus:border-purple-500"
          />
        </div>
      </div>

      {/* Article Content */}
      {contentType === "article" && (
        <div className="space-y-3 bg-white p-3 rounded-lg">
          <h5 className="font-semibold text-gray-900 flex items-center gap-2">
            <FileText className="w-4 h-4" />
            Article Content
          </h5>

          <div>
            <label className="block text-xs font-semibold text-gray-900 mb-1">
              Summary
            </label>
            <input
              type="text"
              value={contentItem.article?.summary || ""}
              onChange={(e) => {
                const article = contentItem.article || {};
                onUpdate(index, {
                  ...contentItem,
                  article: { ...article, summary: e.target.value },
                });
              }}
              placeholder="Brief summary"
              className="w-full px-3 py-2 border border-gray-200 rounded-lg focus:outline-none focus:border-purple-500 text-sm"
            />
          </div>

          <div>
            <label className="block text-xs font-semibold text-gray-900 mb-1">
              Body *
            </label>
            <textarea
              value={contentItem.article?.body || ""}
              onChange={(e) => {
                const article = contentItem.article || {};
                onUpdate(index, {
                  ...contentItem,
                  article: { ...article, body: e.target.value },
                });
              }}
              placeholder="Write your article content..."
              rows="4"
              className="w-full px-3 py-2 border border-gray-200 rounded-lg focus:outline-none focus:border-purple-500 text-sm font-mono"
            />
          </div>
        </div>
      )}

      {/* Quiz Content */}
      {contentType === "quiz" && (
        <div className="space-y-3 bg-white p-3 rounded-lg">
          <h5 className="font-semibold text-gray-900 flex items-center gap-2">
            <HelpCircle className="w-4 h-4" />
            Quiz Settings
          </h5>

          <div className="grid grid-cols-2 gap-3">
            <div>
              <label className="block text-xs font-semibold text-gray-900 mb-1">
                Total Marks *
              </label>
              <input
                type="number"
                value={contentItem.quiz?.total_marks || 10}
                onChange={(e) => {
                  const quiz = contentItem.quiz || {};
                  onUpdate(index, {
                    ...contentItem,
                    quiz: {
                      ...quiz,
                      total_marks: parseInt(e.target.value) || 10,
                    },
                  });
                }}
                min="1"
                className="w-full px-3 py-2 border border-gray-200 rounded-lg focus:outline-none focus:border-purple-500 text-sm"
              />
            </div>

            <div>
              <label className="block text-xs font-semibold text-gray-900 mb-1">
                Passing Marks *
              </label>
              <input
                type="number"
                value={contentItem.quiz?.passing_marks || 5}
                onChange={(e) => {
                  const quiz = contentItem.quiz || {};
                  onUpdate(index, {
                    ...contentItem,
                    quiz: {
                      ...quiz,
                      passing_marks: parseInt(e.target.value) || 5,
                    },
                  });
                }}
                min="1"
                className="w-full px-3 py-2 border border-gray-200 rounded-lg focus:outline-none focus:border-purple-500 text-sm"
              />
            </div>
          </div>

          {/* Quiz Questions */}
          <div className="mt-4">
            <div className="flex items-center justify-between mb-3">
              <h6 className="font-semibold text-gray-900">
                Questions ({contentItem.quiz?.questions?.length || 0})
              </h6>
              <button
                onClick={() => {
                  const quiz = contentItem.quiz || {};
                  const questions = quiz.questions || [];
                  onUpdate(index, {
                    ...contentItem,
                    quiz: {
                      ...quiz,
                      questions: [
                        ...questions,
                        {
                          question_text: "",
                          options: [
                            { text: "", is_correct: true },
                            { text: "", is_correct: false },
                          ],
                        },
                      ],
                    },
                  });
                }}
                className="px-3 py-1.5 bg-blue-600 text-white rounded-lg text-xs font-semibold hover:bg-blue-700 flex items-center gap-1"
              >
                <Plus className="w-3 h-3" />
                Add Question
              </button>
            </div>

            <div className="space-y-2">
              {(contentItem.quiz?.questions || []).map((question, qIdx) => (
                <QuizQuestionEditor
                  key={qIdx}
                  question={question}
                  questionIndex={qIdx}
                  onUpdateQuestion={(qIdx, updatedQ) => {
                    const quiz = contentItem.quiz || {};
                    const questions = [...(quiz.questions || [])];
                    questions[qIdx] = updatedQ;
                    onUpdate(index, {
                      ...contentItem,
                      quiz: { ...quiz, questions },
                    });
                  }}
                  onDeleteQuestion={(qIdx) => {
                    const quiz = contentItem.quiz || {};
                    const questions = quiz.questions.filter(
                      (_, i) => i !== qIdx,
                    );
                    onUpdate(index, {
                      ...contentItem,
                      quiz: { ...quiz, questions },
                    });
                  }}
                />
              ))}
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

// ─── Step 3: Modules with Content Items & Learning Objectives ─────────────────

function Step3Modules({
  data,
  onChange,
  onAddModule,
  onDeleteModule,
  onEditModule,
  onAddContentItem,
  onDeleteContentItem,
  onUpdateContentItem,
  onUpdateLearningObjectives,
}) {
  const [editingModuleId, setEditingModuleId] = useState(null);
  const [newModuleTitle, setNewModuleTitle] = useState("");
  const [newModuleDesc, setNewModuleDesc] = useState("");
  const [expandedModuleId, setExpandedModuleId] = useState(null);

  const handleAddModule = () => {
    if (!newModuleTitle.trim()) return;

    const newModule = {
      id: Date.now(),
      title: newModuleTitle,
      description: newModuleDesc,
      module_order: data.modules.length + 1,
      learning_objectives: ["", "", "", ""],
      content_items: [],
    };

    onAddModule(newModule);
    setNewModuleTitle("");
    setNewModuleDesc("");
  };

  const handleEditModule = (moduleId, newTitle, newDesc) => {
    onEditModule(moduleId, newTitle, newDesc);
    setEditingModuleId(null);
  };

  return (
    <div className="space-y-6">
      <div className="bg-blue-50 border border-blue-200 rounded-xl p-4 flex items-start gap-3">
        <AlertCircle className="w-5 h-5 text-blue-600 flex-shrink-0 mt-0.5" />
        <div>
          <h3 className="font-semibold text-blue-900">Create Course Modules</h3>
          <p className="text-sm text-blue-800 mt-1">
            Add modules with learning objectives and content items (articles &
            quizzes) to structure your course
          </p>
        </div>
      </div>

      {/* Add New Module */}
      <div className="bg-white border border-gray-200 rounded-xl p-4 space-y-3">
        <input
          type="text"
          value={newModuleTitle}
          onChange={(e) => setNewModuleTitle(e.target.value)}
          placeholder="Module title (e.g., DSA Interview Preparation)"
          className="w-full px-4 py-2.5 border border-gray-200 rounded-lg focus:outline-none focus:border-blue-500 focus:ring-2 focus:ring-blue-50"
        />
        <textarea
          value={newModuleDesc}
          onChange={(e) => setNewModuleDesc(e.target.value)}
          placeholder="Brief description of this module (optional)"
          rows="2"
          className="w-full px-4 py-2.5 border border-gray-200 rounded-lg focus:outline-none focus:border-blue-500 focus:ring-2 focus:ring-blue-50"
        />
        <button
          onClick={handleAddModule}
          disabled={!newModuleTitle.trim()}
          className="w-full px-4 py-2.5 bg-blue-600 text-white rounded-lg font-semibold hover:bg-blue-700 disabled:opacity-50 transition flex items-center justify-center gap-2"
        >
          <Plus className="w-4 h-4" />
          Add Module
        </button>
      </div>

      {/* Modules List */}
      {data.modules.length > 0 && (
        <div className="space-y-3">
          <h3 className="font-semibold text-gray-900">
            Modules ({data.modules.length})
          </h3>
          {data.modules.map((module, mIdx) => (
            <div
              key={module.id}
              className="bg-white border border-gray-200 rounded-xl overflow-hidden"
            >
              {/* Module Header */}
              <div className="p-4 border-b border-gray-100">
                {editingModuleId === module.id ? (
                  <div className="space-y-2">
                    <input
                      type="text"
                      defaultValue={module.title}
                      id={`module-title-${module.id}`}
                      className="w-full px-3 py-2 border border-gray-200 rounded-lg focus:outline-none focus:border-blue-500"
                    />
                    <textarea
                      defaultValue={module.description}
                      id={`module-desc-${module.id}`}
                      rows="2"
                      className="w-full px-3 py-2 border border-gray-200 rounded-lg focus:outline-none focus:border-blue-500"
                    />
                    <div className="flex gap-2">
                      <button
                        onClick={() => {
                          const title = document.getElementById(
                            `module-title-${module.id}`,
                          ).value;
                          const desc = document.getElementById(
                            `module-desc-${module.id}`,
                          ).value;
                          handleEditModule(module.id, title, desc);
                        }}
                        className="flex-1 px-3 py-1.5 bg-blue-600 text-white rounded-lg text-sm font-semibold hover:bg-blue-700"
                      >
                        Save
                      </button>
                      <button
                        onClick={() => setEditingModuleId(null)}
                        className="flex-1 px-3 py-1.5 border border-gray-200 text-gray-700 rounded-lg text-sm font-semibold hover:bg-gray-50"
                      >
                        Cancel
                      </button>
                    </div>
                  </div>
                ) : (
                  <div className="flex items-start justify-between">
                    <button
                      onClick={() =>
                        setExpandedModuleId(
                          expandedModuleId === module.id ? null : module.id,
                        )
                      }
                      className="flex-1 text-left hover:opacity-70 transition flex items-start gap-3"
                    >
                      <div className="flex items-center justify-center w-6 h-6 bg-blue-100 text-blue-700 rounded-full text-xs font-bold flex-shrink-0 mt-0.5">
                        {mIdx + 1}
                      </div>
                      <div className="flex-1 min-w-0">
                        <h4 className="font-semibold text-gray-900">
                          {module.title}
                        </h4>
                        {module.description && (
                          <p className="text-sm text-gray-600 mt-1">
                            {module.description}
                          </p>
                        )}
                        <p className="text-xs text-gray-500 mt-1">
                          {module.content_items?.length || 0} content items
                        </p>
                      </div>
                    </button>
                    <div className="flex gap-2 flex-shrink-0">
                      <button
                        onClick={() => setEditingModuleId(module.id)}
                        className="p-2 text-gray-600 hover:bg-gray-100 rounded-lg transition"
                        title="Edit module"
                      >
                        <BookOpen className="w-4 h-4" />
                      </button>
                      <button
                        onClick={() => onDeleteModule(module.id)}
                        className="p-2 text-red-600 hover:bg-red-50 rounded-lg transition"
                        title="Delete module"
                      >
                        <Trash2 className="w-4 h-4" />
                      </button>
                    </div>
                  </div>
                )}
              </div>

              {/* Module Content - Expanded View */}
              {expandedModuleId === module.id && (
                <div className="bg-gray-50 space-y-4 border-t border-gray-100">
                  {/* Learning Objectives */}
                  <div className="p-4 border-b border-gray-200">
                    <h5 className="font-semibold text-gray-900 mb-3 flex items-center gap-2">
                      <Target className="w-4 h-4 text-blue-600" />
                      Learning Objectives/Outcomes
                    </h5>
                    <p className="text-xs text-gray-500 mb-3">
                      What will students learn in this module?
                    </p>
                    <div className="space-y-2">
                      {(module.learning_objectives || ["", "", "", ""]).map(
                        (objective, idx) => (
                          <input
                            key={idx}
                            type="text"
                            value={objective}
                            onChange={(e) => {
                              const newObjectives = [
                                ...(module.learning_objectives || []),
                              ];
                              newObjectives[idx] = e.target.value;
                              onUpdateLearningObjectives(
                                module.id,
                                newObjectives,
                              );
                            }}
                            placeholder={`Learning Objective ${idx + 1} (e.g., Solve common interview problems)`}
                            className="w-full px-3 py-2 border border-gray-200 rounded-lg focus:outline-none focus:border-blue-500 text-sm"
                          />
                        ),
                      )}
                    </div>
                  </div>

                  {/* Content Items */}
                  <div className="p-4">
                    <div className="flex items-center justify-between mb-3">
                      <h5 className="font-semibold text-gray-900">
                        Content Items
                      </h5>
                      <button
                        onClick={() =>
                          onAddContentItem(module.id, {
                            id: Date.now(),
                            title: "",
                            content_type: "article",
                            estimated_duration_minutes: 10,
                            article: {
                              title: "",
                              body: "",
                              summary: "",
                            },
                            quiz: null,
                          })
                        }
                        className="px-3 py-1.5 bg-purple-600 text-white rounded-lg text-xs font-semibold hover:bg-purple-700 flex items-center gap-1"
                      >
                        <Plus className="w-3 h-3" />
                        Add Content
                      </button>
                    </div>

                    {module.content_items && module.content_items.length > 0 ? (
                      <div className="space-y-3">
                        {module.content_items.map((contentItem, cIdx) => (
                          <ContentItemSection
                            key={contentItem.id}
                            contentItem={contentItem}
                            index={cIdx}
                            contentIndex={cIdx}
                            onUpdate={(cIdx, updatedItem) => {
                              onUpdateContentItem(module.id, cIdx, updatedItem);
                            }}
                            onDelete={(cIdx) => {
                              onDeleteContentItem(module.id, cIdx);
                            }}
                          />
                        ))}
                      </div>
                    ) : (
                      <div className="text-center py-6 bg-white rounded-lg border border-dashed border-gray-300">
                        <BookOpen className="w-8 h-8 text-gray-300 mx-auto mb-2" />
                        <p className="text-gray-500 text-sm">
                          No content items yet
                        </p>
                      </div>
                    )}
                  </div>
                </div>
              )}
            </div>
          ))}
        </div>
      )}

      {data.modules.length === 0 && (
        <div className="text-center py-12 bg-white rounded-xl border border-dashed border-gray-300">
          <BookOpen className="w-12 h-12 text-gray-300 mx-auto mb-3" />
          <p className="text-gray-500">No modules added yet</p>
        </div>
      )}
    </div>
  );
}

// ─── Step 4: Review ───────────────────────────────────────────────────────────

function Step4Review({ data }) {
  return (
    <div className="space-y-6">
      <div className="bg-blue-50 border border-blue-200 rounded-xl p-4 flex items-start gap-3">
        <Eye className="w-5 h-5 text-blue-600 flex-shrink-0 mt-0.5" />
        <div>
          <h3 className="font-semibold text-blue-900">Review your course</h3>
          <p className="text-sm text-blue-800 mt-1">
            Verify all details before publishing
          </p>
        </div>
      </div>

      {/* Course Summary */}
      <div className="bg-white border border-gray-200 rounded-xl p-6 space-y-4">
        <div>
          <h2 className="text-2xl font-bold text-gray-900 mb-2">
            {data.title}
          </h2>
          <p className="text-gray-600">{data.description}</p>
        </div>

        <div className="grid grid-cols-3 gap-4 py-4 border-y border-gray-200">
          <div>
            <p className="text-xs text-gray-500 mb-1">Difficulty</p>
            <p className="font-semibold text-gray-900 capitalize">
              {data.difficulty_level}
            </p>
          </div>
          <div>
            <p className="text-xs text-gray-500 mb-1">Price</p>
            <p className="font-semibold text-gray-900">${data.price}</p>
          </div>
          <div>
            <p className="text-xs text-gray-500 mb-1">Modules</p>
            <p className="font-semibold text-gray-900">{data.modules.length}</p>
          </div>
        </div>

        {/* Topics */}
        {data.topics.filter((t) => t).length > 0 && (
          <div>
            <h3 className="font-semibold text-gray-900 mb-2">Topics Covered</h3>
            <div className="flex flex-wrap gap-2">
              {data.topics
                .filter((t) => t)
                .map((topic, idx) => (
                  <span
                    key={idx}
                    className="px-3 py-1 bg-blue-100 text-blue-700 rounded-full text-sm font-medium"
                  >
                    {topic}
                  </span>
                ))}
            </div>
          </div>
        )}

        {/* Course Structure */}
        {data.modules.length > 0 && (
          <div>
            <h3 className="font-semibold text-gray-900 mb-3">
              Course Structure
            </h3>
            <div className="space-y-3">
              {data.modules.map((module, mIdx) => (
                <div
                  key={module.id}
                  className="border border-gray-200 rounded-lg p-3"
                >
                  <div className="flex items-start gap-3 mb-2">
                    <span className="inline-flex items-center justify-center w-6 h-6 bg-blue-100 text-blue-700 rounded-full text-xs font-bold flex-shrink-0">
                      {mIdx + 1}
                    </span>
                    <div className="flex-1">
                      <p className="font-medium text-gray-900">
                        {module.title}
                      </p>
                      {module.description && (
                        <p className="text-xs text-gray-600 mt-1">
                          {module.description}
                        </p>
                      )}
                    </div>
                  </div>

                  {/* Learning Objectives */}
                  {module.learning_objectives?.some((obj) => obj) && (
                    <div className="ml-9 mb-3 pt-2 border-t border-gray-200">
                      <p className="text-xs font-semibold text-gray-700 mb-1 flex items-center gap-1">
                        <Target className="w-3 h-3" />
                        Learning Objectives:
                      </p>
                      <ul className="text-xs text-gray-600 space-y-1">
                        {module.learning_objectives
                          .filter((obj) => obj)
                          .map((obj, idx) => (
                            <li key={idx} className="flex items-start gap-2">
                              <span className="text-blue-600 mt-0.5">•</span>
                              <span>{obj}</span>
                            </li>
                          ))}
                      </ul>
                    </div>
                  )}

                  {/* Content Items */}
                  {module.content_items && module.content_items.length > 0 && (
                    <div className="ml-9 space-y-1 pt-2 border-t border-gray-200">
                      <p className="text-xs font-semibold text-gray-700 mb-1">
                        Content:
                      </p>
                      {module.content_items.map((item, cIdx) => (
                        <div
                          key={item.id}
                          className="text-xs text-gray-600 flex items-center gap-2"
                        >
                          {item.content_type === "article" ? (
                            <>
                              <FileText className="w-3 h-3" />
                              <span>{item.title} (Article)</span>
                            </>
                          ) : (
                            <>
                              <HelpCircle className="w-3 h-3" />
                              <span>
                                {item.title} (Quiz -{" "}
                                {item.quiz?.questions?.length || 0} Q's)
                              </span>
                            </>
                          )}
                        </div>
                      ))}
                    </div>
                  )}
                </div>
              ))}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

// ─── Main Create Course Page ───────────────────────────────────────────────────

export default function CreateCoursePage() {
  const navigate = useNavigate();
  const [currentStep, setCurrentStep] = useState(0);
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [error, setError] = useState("");
  const [courseImage, setCourseImage] = useState(null);

  const [formData, setFormData] = useState({
    title: "",
    description: "",
    difficulty_level: "intermediate",
    price: "1799.00",
    course_type: "instructor_published",
    topics: [""],
    modules: [],
  });

  const steps = [
    { title: "Basic Info", component: Step1BasicInfo },
    { title: "Topics & Image", component: Step2ImageAndTopics },
    { title: "Modules & Content", component: Step3Modules },
    { title: "Review", component: Step4Review },
  ];

  const handleInputChange = (e) => {
    const { name, value } = e.target;
    setFormData((prev) => ({
      ...prev,
      [name]: value,
    }));
  };

  const handleImageUpload = (file) => {
    setCourseImage(file);
  };

  const handleAddModule = (newModule) => {
    setFormData((prev) => ({
      ...prev,
      modules: [...prev.modules, newModule],
    }));
  };

  const handleDeleteModule = (moduleId) => {
    setFormData((prev) => ({
      ...prev,
      modules: prev.modules.filter((m) => m.id !== moduleId),
    }));
  };

  const handleEditModule = (moduleId, newTitle, newDesc) => {
    setFormData((prev) => ({
      ...prev,
      modules: prev.modules.map((m) =>
        m.id === moduleId ? { ...m, title: newTitle, description: newDesc } : m,
      ),
    }));
  };

  const handleUpdateLearningObjectives = (moduleId, objectives) => {
    setFormData((prev) => ({
      ...prev,
      modules: prev.modules.map((m) =>
        m.id === moduleId ? { ...m, learning_objectives: objectives } : m,
      ),
    }));
  };

  const handleAddContentItem = (moduleId, newContentItem) => {
    setFormData((prev) => ({
      ...prev,
      modules: prev.modules.map((m) =>
        m.id === moduleId
          ? {
              ...m,
              content_items: [...(m.content_items || []), newContentItem],
            }
          : m,
      ),
    }));
  };

  const handleDeleteContentItem = (moduleId, contentIndex) => {
    setFormData((prev) => ({
      ...prev,
      modules: prev.modules.map((m) =>
        m.id === moduleId
          ? {
              ...m,
              content_items: m.content_items.filter(
                (_, i) => i !== contentIndex,
              ),
            }
          : m,
      ),
    }));
  };

  const handleUpdateContentItem = (moduleId, contentIndex, updatedItem) => {
    setFormData((prev) => ({
      ...prev,
      modules: prev.modules.map((m) =>
        m.id === moduleId
          ? {
              ...m,
              content_items: m.content_items.map((item, i) =>
                i === contentIndex ? updatedItem : item,
              ),
            }
          : m,
      ),
    }));
  };

  const isStepValid = () => {
    switch (currentStep) {
      case 0:
        return (
          formData.title &&
          formData.description &&
          formData.difficulty_level &&
          formData.price
        );
      case 1:
        return formData.topics.some((t) => t.trim());
      case 2:
        return (
          formData.modules.length > 0 &&
          formData.modules.some((m) => m.content_items?.length > 0)
        );
      case 3:
        return true;
      default:
        return false;
    }
  };

  const handleNext = () => {
    if (isStepValid()) {
      setCurrentStep((prev) => Math.min(prev + 1, steps.length - 1));
      window.scrollTo({ top: 0, behavior: "smooth" });
    }
  };

  const handlePrevious = () => {
    setCurrentStep((prev) => Math.max(prev - 1, 0));
    window.scrollTo({ top: 0, behavior: "smooth" });
  };

  const handleSubmit = async () => {
    if (!isStepValid()) {
      setError("Please fill in all required fields");
      return;
    }

    try {
      setIsSubmitting(true);
      setError("");

      const token = localStorage.getItem("access_token");
      if (!token) {
        navigate("/login");
        return;
      }

      // Step 1: Create the course
      const coursePayload = {
        title: formData.title,
        description: formData.description,
        difficulty_level: formData.difficulty_level,
        price: formData.price,
        course_type: formData.course_type,
        topics: formData.topics.filter((t) => t.trim()),
        is_published: true,
      };

      const courseRes = await fetch(`${API_URL}/api/courses/courses/`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          Authorization: `Bearer ${token}`,
        },
        body: JSON.stringify(coursePayload),
      });

      if (!courseRes.ok) {
        throw new Error(`Failed to create course: ${courseRes.status}`);
      }

      const course = await courseRes.json();
      const courseId = course.id;

      // Step 2: Create modules and content items
      for (const module of formData.modules) {
        const modulePayload = {
          course: courseId,
          title: module.title,
          description: module.description,
          module_order: module.module_order,
          learning_objectives: module.learning_objectives
            .filter((obj) => obj.trim())
            .map((obj) => obj.trim()),
        };

        const moduleRes = await fetch(
          `${API_URL}/api/courses/course-modules/`,
          {
            method: "POST",
            headers: {
              "Content-Type": "application/json",
              Authorization: `Bearer ${token}`,
            },
            body: JSON.stringify(modulePayload),
          },
        );

        if (!moduleRes.ok) {
          throw new Error(`Failed to create module: ${moduleRes.status}`);
        }

        const createdModule = await moduleRes.json();
        const moduleId = createdModule.id;

        // Step 3: Create content items for this module
        if (module.content_items && module.content_items.length > 0) {
          for (let cIdx = 0; cIdx < module.content_items.length; cIdx++) {
            const contentItem = module.content_items[cIdx];

            // Create content item
            const contentPayload = {
              module: moduleId,
              title: contentItem.title,
              content_type: contentItem.content_type,
              difficulty: contentItem.difficulty || "easy",
              estimated_duration_minutes:
                contentItem.estimated_duration_minutes,
              order: cIdx + 1,
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
              throw new Error(
                `Failed to create content item: ${contentRes.status}`,
              );
            }

            const createdContent = await contentRes.json();
            const contentItemId = createdContent.id;

            // Step 4a: Create article if it's an article
            if (contentItem.content_type === "article" && contentItem.article) {
              const articlePayload = {
                content_item: contentItemId,
                body: contentItem.article.body,
                summary: contentItem.article.summary || "",
              };

              const articleRes = await fetch(
                `${API_URL}/api/courses/articles/`,
                {
                  method: "POST",
                  headers: {
                    "Content-Type": "application/json",
                    Authorization: `Bearer ${token}`,
                  },
                  body: JSON.stringify(articlePayload),
                },
              );

              if (!articleRes.ok) {
                throw new Error(
                  `Failed to create article: ${articleRes.status}`,
                );
              }
            }

            // Step 4b: Create quiz if it's a quiz
            if (contentItem.content_type === "quiz" && contentItem.quiz) {
              const quizPayload = {
                content_item: contentItemId,
                total_marks: contentItem.quiz.total_marks,
                passing_marks: contentItem.quiz.passing_marks,
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

              // Create quiz questions and options
              if (
                contentItem.quiz.questions &&
                contentItem.quiz.questions.length > 0
              ) {
                for (
                  let qIdx = 0;
                  qIdx < contentItem.quiz.questions.length;
                  qIdx++
                ) {
                  const question = contentItem.quiz.questions[qIdx];

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
                    throw new Error(
                      `Failed to create question: ${questionRes.status}`,
                    );
                  }

                  const createdQuestion = await questionRes.json();
                  const questionId = createdQuestion.id;

                  // Create options for this question
                  if (question.options && question.options.length > 0) {
                    for (let oIdx = 0; oIdx < question.options.length; oIdx++) {
                      const option = question.options[oIdx];

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
                        throw new Error(
                          `Failed to create option: ${optionRes.status}`,
                        );
                      }
                    }
                  }
                }
              }
            }
          }
        }
      }

      navigate(`/instructor/course/${courseId}`, {
        state: { message: "Course created successfully with all content!" },
      });
    } catch (err) {
      console.error("Error:", err);
      setError(err instanceof Error ? err.message : "Failed to create course");
    } finally {
      setIsSubmitting(false);
    }
  };

  const CurrentStepComponent = steps[currentStep].component;

  return (
    <div className="min-h-screen bg-gray-50">
      {/* Header */}
      <div className="bg-white shadow-sm border-b sticky top-0 z-40">
        <div className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8 py-6">
          <button
            onClick={() => navigate(-1)}
            className="inline-flex items-center gap-2 text-gray-600 hover:text-blue-600 transition mb-4"
          >
            <ArrowLeft className="w-4 h-4" /> Back
          </button>
          <h1 className="text-2xl font-bold text-gray-900">
            Create New Course
          </h1>
          <p className="text-sm text-gray-600 mt-1">
            Step {currentStep + 1} of {steps.length}: {steps[currentStep].title}
          </p>
        </div>
      </div>

      {/* Main Content */}
      <div className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {/* Step Indicator */}
        <StepIndicator currentStep={currentStep} totalSteps={steps.length} />

        {/* Error Message */}
        {error && (
          <div className="mb-6 p-4 bg-red-50 border border-red-200 rounded-xl text-red-700 flex items-start gap-3">
            <AlertCircle className="w-5 h-5 flex-shrink-0 mt-0.5" />
            <div>
              <p className="font-semibold">Error</p>
              <p className="text-sm mt-1">{error}</p>
            </div>
          </div>
        )}

        {/* Step Content */}
        <div className="bg-white rounded-2xl shadow-sm border border-gray-100 p-8 mb-8">
          <CurrentStepComponent
            data={formData}
            onChange={handleInputChange}
            onImageUpload={handleImageUpload}
            onAddModule={handleAddModule}
            onDeleteModule={handleDeleteModule}
            onEditModule={handleEditModule}
            onUpdateLearningObjectives={handleUpdateLearningObjectives}
            onAddContentItem={handleAddContentItem}
            onDeleteContentItem={handleDeleteContentItem}
            onUpdateContentItem={handleUpdateContentItem}
          />
        </div>

        {/* Navigation Buttons */}
        <div className="flex gap-4">
          <button
            onClick={handlePrevious}
            disabled={currentStep === 0}
            className="px-6 py-3 border border-gray-200 text-gray-700 rounded-xl font-semibold hover:bg-gray-50 disabled:opacity-50 disabled:cursor-not-allowed transition flex items-center gap-2"
          >
            <ChevronLeft className="w-4 h-4" /> Previous
          </button>

          {currentStep === steps.length - 1 ? (
            <button
              onClick={handleSubmit}
              disabled={isSubmitting || !isStepValid()}
              className="flex-1 px-6 py-3 bg-blue-600 text-white rounded-xl font-semibold hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed transition flex items-center justify-center gap-2"
            >
              <Save className="w-4 h-4" />
              {isSubmitting ? "Creating Course..." : "Create Course"}
            </button>
          ) : (
            <button
              onClick={handleNext}
              disabled={!isStepValid()}
              className="flex-1 px-6 py-3 bg-blue-600 text-white rounded-xl font-semibold hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed transition flex items-center justify-center gap-2"
            >
              Next <ChevronRight className="w-4 h-4" />
            </button>
          )}
        </div>
      </div>
    </div>
  );
}
