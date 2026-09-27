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
import { useState } from "react";
import { useNavigate } from "react-router-dom";

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

export default function ContentItemCard({ item, courseId, moduleId }) {
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
