import { useQuery } from "@tanstack/react-query";
import { useState } from "react";
import { getCourseModules } from "../services/courseDetail";
import {
  Clock,
  ChevronUp,
  ChevronDown,
  Layers,
  CheckCircle,
} from "lucide-react";
import ContentItemCard from "./ContentItemCard";

export default function ModuleCard({ moduleData, index, courseId }) {
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
    <div className="bg-white border border-gray-100 rounded-xl shadow-sm hover:shadow-md transition-all overflow-hidden">
      {/* Module Header */}
      <button
        onClick={handleOpen}
        className="w-full flex items-center gap-4 p-5 text-left hover:bg-gray-50 transition-colors"
      >
        <div className="w-10 h-10 bg-gradient-to-br from-blue-500 to-purple-600 rounded-xl flex items-center justify-center text-white font-bold text-sm flex-shrink-0">
          {String(index + 1).padStart(2, "0")}
        </div>
        <div className="flex-1 min-w-0">
          <p className="font-bold text-gray-900 text-sm">{displayTitle}</p>
          <p className="text-xs text-gray-500 line-clamp-1 mt-0.5">
            {displayDescription}
          </p>
        </div>
        <div className="flex items-center gap-3 flex-shrink-0">
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
                    {displayDuration}
                  </span>
                )}
              </>
            )}
            {isLoading && (
              <span className="text-xs text-gray-400">Loading...</span>
            )}
          </div>
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
          <p className="text-sm text-red-700">Error loading module details.</p>
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
          )}
        </div>
      )}
    </div>
  );
}
