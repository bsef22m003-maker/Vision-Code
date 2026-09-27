import { useQuery } from "@tanstack/react-query";
import { useState } from "react";
import { getCourseModules } from "../services/courseDetail";
import { formatDuration } from "../utils/util";
import {
  ArrowLeft,
  BookOpen,
  HelpCircle,
  Code,
  Star,
  Clock,
  Users,
  Trophy,
  PlayCircle,
  CheckCircle,
  ChevronDown,
  ChevronUp,
  Zap,
} from "lucide-react";
const contentTypeIcons = {
  article: BookOpen,
  quiz: HelpCircle,
  exercise: Code,
  video: PlayCircle,
};

const contentTypeColors = {
  article: "text-blue-600 bg-blue-50 border-blue-200",
  quiz: "text-purple-600 bg-purple-50 border-purple-200",
  exercise: "text-orange-600 bg-orange-50 border-orange-200",
  video: "text-green-600 bg-green-50 border-green-200",
};
export default function ModuleAccordion({ moduleData, idx }) {
  const [open, setOpen] = useState(false);
  const { data, isLoading, error, refetch } = useQuery({
    queryKey: ["modules", moduleData.id],
    queryFn: () => getCourseModules(moduleData.id),
    enabled: open,
  });
  function handleOpen() {
    setOpen((v) => !v);
    refetch();
  }
  const totalDuration = data?.content_items?.reduce(
    (sum, item) => sum + (item.estimated_duration_minutes || 0),
    0,
  );

  const module = data;

  // Determine which data to show in header
  const displayTitle = module?.title || moduleData.title;
  const displayDescription = module?.description || moduleData.description;
  const displayDuration = module
    ? totalDuration
    : moduleData.estimated_duration || "";
  const displayItemCount =
    module?.content_items?.length || moduleData.content_items?.length || "";
  return (
    <div className="border border-gray-200 rounded-xl overflow-hidden shadow-sm hover:shadow-md transition-shadow duration-200">
      {/* Accordion Header */}
      <button
        onClick={handleOpen}
        className="w-full flex items-center gap-4 p-4 bg-white hover:bg-gray-50 transition-colors text-left"
      >
        <div className="w-9 h-9 flex-shrink-0 rounded-lg bg-gradient-to-br from-blue-500 to-purple-600 flex items-center justify-center text-white font-bold text-sm">
          {String(idx + 1).padStart(2, "0")}
        </div>
        <div className="flex-1 min-w-0">
          <p className="font-semibold text-gray-900 text-sm truncate">
            {displayTitle}
          </p>
          <p className="text-xs text-gray-500 truncate">{displayDescription}</p>
        </div>
        <div className="flex items-center gap-3 flex-shrink-0">
          {!isLoading && (
            <>
              {displayDuration && (
                <span className="text-xs text-gray-400 flex items-center gap-1">
                  <Clock className="w-3 h-3" />
                  {formatDuration(displayDuration)}
                </span>
              )}
              {displayItemCount && (
                <span className="text-xs text-gray-400">
                  {displayItemCount} items
                </span>
              )}
            </>
          )}
          {isLoading && (
            <span className="text-xs text-gray-400">Loading...</span>
          )}
          {open ? (
            <ChevronUp className="w-4 h-4 text-gray-400" />
          ) : (
            <ChevronDown className="w-4 h-4 text-gray-400" />
          )}
        </div>
      </button>

      {/* Accordion Body */}
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
          <p className="text-sm text-red-700">Error loading module details.</p>
        </div>
      )}

      {open && module && (
        <div className="border-t border-gray-100 bg-gray-50 p-4 space-y-4">
          {/* Learning Objectives */}
          {module.learning_objectives?.length > 0 && (
            <div>
              <p className="text-xs font-semibold text-gray-500 uppercase tracking-wider mb-2">
                What you'll learn
              </p>
              <ul className="space-y-1.5">
                {module.learning_objectives.map((obj, i) => (
                  <li
                    key={i}
                    className="flex items-start gap-2 text-sm text-gray-700"
                  >
                    <CheckCircle className="w-3.5 h-3.5 text-green-500 mt-0.5 flex-shrink-0" />
                    {typeof obj === "string"
                      ? obj
                      : (obj?.title ?? JSON.stringify(obj))}
                  </li>
                ))}
              </ul>
            </div>
          )}

          {/* Content Items */}
          {module.content_items?.length > 0 && (
            <div>
              <p className="text-xs font-semibold text-gray-500 uppercase tracking-wider mb-2">
                Content
              </p>
              <div className="space-y-2">
                {module.content_items.map((item) => {
                  const Icon = contentTypeIcons[item.content_type] || BookOpen;
                  const colorClass =
                    contentTypeColors[item.content_type] ||
                    "text-gray-600 bg-gray-50 border-gray-200";
                  return (
                    <div
                      key={item.id}
                      className="flex items-center gap-3 p-2.5 bg-white rounded-lg border border-gray-200"
                    >
                      <div className={`p-1.5 rounded-md border ${colorClass}`}>
                        <Icon className="w-3.5 h-3.5" />
                      </div>
                      <span className="flex-1 text-sm text-gray-800 truncate">
                        {item.title}
                      </span>
                      <span
                        className={`text-xs px-2 py-0.5 rounded-full border font-medium ${colorClass}`}
                      >
                        {item.content_type}
                      </span>
                      <span className="text-xs text-gray-400 flex items-center gap-1 flex-shrink-0">
                        <Clock className="w-3 h-3" />
                        {item.estimated_duration_minutes}m
                      </span>
                    </div>
                  );
                })}
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
