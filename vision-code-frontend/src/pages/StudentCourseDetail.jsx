import { useState } from "react";
import { useParams, Link, useNavigate } from "react-router-dom";
import {
  ArrowLeft,
  BookOpen,
  HelpCircle,
  Code,
  Star,
  Clock,
  CheckCircle,
  ChevronDown,
  ChevronUp,
  Play,
  FileText,
  Trophy,
  Users,
  BarChart2,
  Layers,
  Tag,
  LogOut,
  Settings,
} from "lucide-react";

import { useAuth } from "../context/FakeAuth";
import { useCourseDetail } from "../hooks/useCourseDetail";

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

const getDifficultyColor = (level) => {
  switch (level) {
    case "beginner":
      return "bg-green-100 text-green-700 border border-green-200";
    case "intermediate":
      return "bg-purple-100 text-purple-700 border border-purple-200";
    case "advanced":
      return "bg-orange-100 text-orange-700 border border-orange-200";
    default:
      return "bg-gray-100 text-gray-700 border border-gray-200";
  }
};

const getProgressColor = (progress) => {
  if (progress >= 75) return "from-green-500 to-emerald-500";
  if (progress >= 50) return "from-blue-500 to-indigo-500";
  if (progress >= 25) return "from-yellow-500 to-orange-500";
  return "from-red-500 to-pink-500";
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

// ─── Article Expand ───────────────────────────────────────────────────────────

function ArticleExpand({ item }) {
  const [expanded, setExpanded] = useState(false);

  return (
    <div className="mt-3">
      <button
        onClick={() => setExpanded((v) => !v)}
        className="flex items-center gap-2 text-xs font-semibold text-blue-600 bg-blue-50 hover:bg-blue-100 border border-blue-200 px-3 py-1.5 rounded-lg transition-all"
      >
        <FileText className="w-3.5 h-3.5" />
        {expanded ? "Close Article" : "Read Article"}
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
                This article covers the fundamentals of{" "}
                <span className="text-blue-600 font-medium">{item.title}</span>.
                Work through the material at your own pace — estimated reading
                time is{" "}
                <span className="font-medium">
                  {item.estimated_duration_minutes} minutes
                </span>
                .
              </p>
              <p>
                Content will be loaded from the server. If you see this
                placeholder, the{" "}
                <code className="bg-gray-200 px-1 py-0.5 rounded text-xs">
                  content_data
                </code>{" "}
                field has not been populated yet.
              </p>
              <div className="flex items-center gap-2 pt-2 border-t border-gray-200 text-xs text-gray-400">
                <Clock className="w-3.5 h-3.5" />~
                {item.estimated_duration_minutes} min read
                <span className="ml-auto capitalize">
                  {item.difficulty || "standard"} level
                </span>
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
}

// ─── Quiz Button ──────────────────────────────────────────────────────────────

function QuizStartButton({ item, courseId, moduleId }) {
  const navigate = useNavigate();

  return (
    <div className="mt-3">
      <button
        onClick={() =>
          navigate(`/student/quiz/${item.id}`, {
            state: {
              quizData: item.content_data,
              title: item.title,
              courseId,
              moduleId,
              estimatedMinutes: item.estimated_duration_minutes,
            },
          })
        }
        className="flex items-center gap-2 text-xs font-semibold text-purple-600 bg-purple-50 hover:bg-purple-100 border border-purple-200 px-3 py-1.5 rounded-lg transition-all"
      >
        <Play className="w-3.5 h-3.5 fill-purple-500" />
        Start Quiz
        {item.content_data && (
          <span className="bg-purple-200 text-purple-700 px-1.5 py-0.5 rounded-full text-[10px] font-bold">
            {item.content_data.questions?.length || 0}Q
          </span>
        )}
      </button>
    </div>
  );
}

// ─── Content Item Card ────────────────────────────────────────────────────────

function ContentItemCard({ item, courseId, moduleId }) {
  const config =
    contentTypeConfig[item.content_type] || contentTypeConfig.article;
  const Icon = config.icon;

  return (
    <div className="border border-gray-100 rounded-xl p-3 hover:border-blue-200 hover:shadow-sm transition-all bg-white">
      <div className="flex items-center gap-3">
        <div
          className={`w-8 h-8 rounded-lg flex items-center justify-center flex-shrink-0 ${config.badge} border`}
        >
          <Icon className="w-4 h-4" />
        </div>
        <div className="flex-1 min-w-0">
          <p className="text-sm font-semibold text-gray-900 truncate">
            {item.title}
          </p>
          <div className="flex items-center gap-2 mt-0.5">
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

      {item.content_type === "article" && <ArticleExpand item={item} />}
      {item.content_type === "quiz" && (
        <QuizStartButton item={item} courseId={courseId} moduleId={moduleId} />
      )}
    </div>
  );
}

// ─── Module Card ──────────────────────────────────────────────────────────────

function ModuleCard({ module, index, courseId }) {
  const [open, setOpen] = useState(index === 0);

  const totalMinutes = module.content_items.reduce(
    (sum, item) => sum + (item.estimated_duration_minutes || 0),
    0,
  );

  return (
    <div className="bg-white border border-gray-100 rounded-xl shadow-sm hover:shadow-md transition-all overflow-hidden">
      {/* Module Header */}
      <button
        onClick={() => setOpen((v) => !v)}
        className="w-full flex items-center gap-4 p-5 text-left hover:bg-gray-50 transition-colors"
      >
        <div className="w-10 h-10 bg-gradient-to-br from-blue-500 to-purple-600 rounded-xl flex items-center justify-center text-white font-bold text-sm flex-shrink-0">
          {String(index + 1).padStart(2, "0")}
        </div>
        <div className="flex-1 min-w-0">
          <p className="font-bold text-gray-900 text-sm">{module.title}</p>
          <p className="text-xs text-gray-500 line-clamp-1 mt-0.5">
            {module.description}
          </p>
        </div>
        <div className="flex items-center gap-3 flex-shrink-0">
          <div className="hidden sm:flex items-center gap-3 text-xs text-gray-400">
            <span className="flex items-center gap-1">
              <Layers className="w-3.5 h-3.5" />
              {module.content_items.length} items
            </span>
            <span className="flex items-center gap-1">
              <Clock className="w-3.5 h-3.5" />
              {totalMinutes}m
            </span>
          </div>
          {open ? (
            <ChevronUp className="w-4 h-4 text-gray-400" />
          ) : (
            <ChevronDown className="w-4 h-4 text-gray-400" />
          )}
        </div>
      </button>

      {/* Module Body */}
      {open && (
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
          <div>
            <p className="text-xs font-semibold text-gray-500 uppercase tracking-wider mb-2">
              Content ({module.content_items.length})
            </p>
            <div className="space-y-2">
              {module.content_items.map((item) => (
                <ContentItemCard
                  key={item.id}
                  item={item}
                  courseId={courseId}
                  moduleId={module.id}
                />
              ))}
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

// ─── Main Page ────────────────────────────────────────────────────────────────

export default function StudentCourseDetail() {
  const { courseId } = useParams();
  const navigate = useNavigate();
  const id = courseId ? parseInt(courseId) : 0;

  const { data: course, isLoading, error } = useCourseDetail(id);
  const { logout, user } = useAuth();

  // TODO: Replace with real enrollment/progress API data
  const progress = 0;
  const enrolled_at = undefined;
  const completed = false;

  const level = course?.difficulty_level?.toLowerCase();
  const gradient = levelGradients[level] || "from-blue-500 to-indigo-600";
  const icon = levelIcons[level] || "📖";

  const initials = user
    ? `${user.firstName?.[0] ?? ""}${user.lastName?.[0] ?? ""}`.toUpperCase()
    : "S";

  const handleLogout = () => {
    logout();
    navigate("/login");
  };

  // ── Loading ──────────────────────────────────────────────────────────────

  if (isLoading) {
    return (
      <div className="min-h-screen bg-gray-50 flex items-center justify-center">
        <div className="text-center">
          <div className="w-12 h-12 border-4 border-blue-600 border-t-transparent rounded-full animate-spin mx-auto mb-4" />
          <p className="text-gray-500 font-medium">Loading course details...</p>
        </div>
      </div>
    );
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
            to="/student"
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
          to="/student"
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
          <div className="relative z-10 flex items-start gap-6 flex-wrap">
            <div className="w-16 h-16 bg-white/20 rounded-2xl flex items-center justify-center text-3xl flex-shrink-0 shadow-lg backdrop-blur-sm">
              {icon}
            </div>
            <div className="flex-1 min-w-0">
              <div className="flex flex-wrap items-center gap-2 mb-2">
                <span className="text-xs font-semibold bg-white/20 px-2 py-0.5 rounded-full capitalize">
                  {course.difficulty_level}
                </span>
                {completed && (
                  <span className="text-xs font-semibold bg-green-400/30 border border-green-300/50 px-2 py-0.5 rounded-full flex items-center gap-1">
                    <CheckCircle className="w-3 h-3" /> Completed
                  </span>
                )}
              </div>
              <h1 className="text-2xl sm:text-3xl font-bold mb-2">
                {course.title}
              </h1>
              <p className="text-white/80 text-sm max-w-2xl">
                {course.description}
              </p>
              <div className="flex items-center gap-4 mt-3 text-sm text-white/70">
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
          </div>
        </div>

        {/* ── Progress Bar (if enrolled) ── */}
        <div className="bg-white rounded-xl shadow-sm border border-gray-100 p-5 mb-8">
          <div className="flex items-center justify-between mb-2">
            <p className="font-semibold text-gray-900 text-sm">Your Progress</p>
            <span className="text-sm font-bold text-blue-600">{progress}%</span>
          </div>
          <div className="w-full bg-gray-100 rounded-full h-2.5">
            <div
              className={`bg-gradient-to-r ${getProgressColor(progress)} h-2.5 rounded-full transition-all duration-500`}
              style={{ width: `${progress}%` }}
            />
          </div>
          <div className="flex items-center justify-between mt-2 text-xs text-gray-400">
            <span>{completed ? "✅ Course completed!" : "Keep going!"}</span>
            {enrolled_at && (
              <span>Enrolled {new Date(enrolled_at).toLocaleDateString()}</span>
            )}
          </div>
        </div>

        {/* ── Main Grid ── */}
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
          {/* Left — Modules */}
          <div className="lg:col-span-2 space-y-6">
            {/* Instructor */}
            <div className="bg-white rounded-xl shadow-sm border border-gray-100 p-5 flex items-center gap-4">
              <div className="w-12 h-12 bg-gradient-to-br from-blue-500 to-purple-600 rounded-xl flex items-center justify-center text-white font-bold text-lg flex-shrink-0">
                {course.instructor.name?.charAt(0).toUpperCase()}
              </div>
              <div>
                <p className="text-xs text-gray-400 font-medium">Instructor</p>
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
                <span className="text-sm text-gray-400">
                  {course.modules.length} modules
                </span>
              </div>
              <div className="space-y-3">
                {course.modules.map((module, idx) => (
                  <ModuleCard
                    key={module.id}
                    module={module}
                    index={idx}
                    courseId={id}
                  />
                ))}
              </div>
            </div>
          </div>

          {/* Right — Sidebar */}
          <div className="space-y-5">
            {/* Continue CTA */}
            <div className="bg-gradient-to-br from-blue-600 via-purple-600 to-indigo-600 rounded-2xl p-6 text-white relative overflow-hidden">
              <div className="absolute top-0 right-0 w-32 h-32 bg-white/5 rounded-full -translate-y-1/2 translate-x-1/2" />
              <h3 className="font-bold text-lg mb-1 relative z-10">
                Ready to learn?
              </h3>
              <p className="text-blue-100 text-xs mb-4 relative z-10">
                Pick up where you left off and keep building your skills.
              </p>
              <button className="w-full py-2.5 bg-white text-blue-600 rounded-xl font-bold text-sm hover:bg-blue-50 transition relative z-10">
                Continue Learning →
              </button>
            </div>

            {/* Stats */}
            <div className="bg-white rounded-xl shadow-sm border border-gray-100 p-5">
              <h3 className="font-bold text-gray-900 mb-4">Course Info</h3>
              <div className="space-y-3">
                {[
                  {
                    icon: <Layers className="w-4 h-4 text-blue-500" />,
                    label: "Modules",
                    value: course.total_modules,
                    bg: "bg-blue-50",
                  },
                  {
                    icon: <Users className="w-4 h-4 text-purple-500" />,
                    label: "Students",
                    value: course.total_students?.toLocaleString(),
                    bg: "bg-purple-50",
                  },
                  {
                    icon: <Star className="w-4 h-4 text-yellow-500" />,
                    label: "Rating",
                    value: "4.5 / 5",
                    bg: "bg-yellow-50",
                  },
                  {
                    icon: <BarChart2 className="w-4 h-4 text-orange-500" />,
                    label: "Level",
                    value:
                      course.difficulty_level?.charAt(0).toUpperCase() +
                      course.difficulty_level?.slice(1),
                    bg: "bg-orange-50",
                  },
                  {
                    icon: <Trophy className="w-4 h-4 text-green-500" />,
                    label: "Status",
                    value: completed ? "Completed ✅" : "In Progress",
                    bg: "bg-green-50",
                  },
                ].map(({ icon, label, value, bg }) => (
                  <div
                    key={label}
                    className="flex items-center justify-between"
                  >
                    <div className="flex items-center gap-2 text-sm text-gray-500">
                      <div
                        className={`w-7 h-7 ${bg} rounded-lg flex items-center justify-center`}
                      >
                        {icon}
                      </div>
                      {label}
                    </div>
                    <span className="text-sm font-semibold text-gray-900">
                      {value}
                    </span>
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

            {/* Daily Challenge */}
            <div className="bg-gradient-to-br from-orange-500 to-pink-600 rounded-xl p-5 text-white">
              <div className="flex items-center justify-between mb-2">
                <h3 className="font-bold">Daily Challenge</h3>
                <span className="text-2xl">🎯</span>
              </div>
              <p className="text-sm text-orange-50 mb-4">
                Complete today's challenge and earn 50 XP bonus!
              </p>
              <button className="w-full py-2 bg-white/20 hover:bg-white/30 border border-white/30 text-white rounded-lg text-sm font-semibold transition">
                Start Challenge
              </button>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
