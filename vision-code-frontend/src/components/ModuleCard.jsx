import { useState, useEffect } from "react";
import ContentItem from "./ContentItem";

function ModuleCard({ module, index, isExpanded, onToggle, onStartQuiz }) {
  const [contentItems, setContentItems] = useState([]);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    if (isExpanded && contentItems.length === 0) {
      fetchContentItems();
    }
  }, [isExpanded]);

  const fetchContentItems = async () => {
    try {
      setLoading(true);
      const response = await fetch(
        `http://127.0.0.1:8000/api/courses/content-items/?module_id=${module.id}`
      );

      if (!response.ok) {
        throw new Error("Failed to fetch content items");
      }

      const data = await response.json();
      setContentItems(data.results);
      console.log("Fetched content items:", data.results);
    } catch (err) {
      console.error("Error fetching content items:", err);
    } finally {
      setLoading(false);
    }
  };

  const difficultyColors = {
    easy: "bg-green-100 text-green-700",
    medium: "bg-yellow-100 text-yellow-700",
    hard: "bg-red-100 text-red-700",
  };

  return (
    <div className="bg-white rounded-xl shadow-md border border-gray-200 overflow-hidden transition-all duration-300 hover:shadow-lg">
      {/* Module Header */}
      <button
        onClick={onToggle}
        className="w-full p-6 flex items-center justify-between hover:bg-gray-50 transition"
      >
        <div className="flex items-center gap-4 flex-1 text-left">
          {/* Module Number */}
          <div className="w-12 h-12 bg-gradient-to-br from-blue-600 to-purple-600 rounded-lg flex items-center justify-center text-white font-bold text-lg flex-shrink-0">
            {module.module_order}
          </div>

          {/* Module Info */}
          <div className="flex-1">
            <h3 className="text-xl font-bold text-gray-900 mb-1">
              {module.title}
            </h3>
            <p className="text-gray-600 text-sm">{module.description}</p>
          </div>
        </div>

        {/* Expand Icon */}
        <svg
          className={`w-6 h-6 text-gray-400 transition-transform duration-300 ${
            isExpanded ? "rotate-180" : ""
          }`}
          fill="none"
          stroke="currentColor"
          viewBox="0 0 24 24"
        >
          <path
            strokeLinecap="round"
            strokeLinejoin="round"
            strokeWidth="2"
            d="M19 9l-7 7-7-7"
          />
        </svg>
      </button>

      {/* Expanded Content */}
      {isExpanded && (
        <div className="border-t border-gray-200 bg-gray-50 p-6 space-y-4">
          {/* Learning Objectives */}
          <div className="mb-6">
            <h4 className="font-semibold text-gray-900 mb-3 flex items-center gap-2">
              <svg
                className="w-5 h-5 text-blue-600"
                fill="none"
                stroke="currentColor"
                viewBox="0 0 24 24"
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth="2"
                  d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z"
                />
              </svg>
              Learning Objectives
            </h4>
            <ul className="space-y-2">
              {module.learning_objectives.map((objective, idx) => (
                <li
                  key={idx}
                  className="flex items-start gap-2 text-gray-700 text-sm"
                >
                  <span className="text-blue-600 mt-1">•</span>
                  <span>{objective}</span>
                </li>
              ))}
            </ul>
          </div>

          {/* Content Items */}
          {loading ? (
            <div className="text-center py-8">
              <div className="animate-spin rounded-full h-8 w-8 border-t-2 border-b-2 border-blue-600 mx-auto"></div>
              <p className="text-gray-600 text-sm mt-2">Loading content... </p>
            </div>
          ) : (
            <div className="space-y-3">
              <h4 className="font-semibold text-gray-900 mb-3">
                Module Content
              </h4>
              {contentItems.map((item) => (
                <ContentItem
                  key={item.id}
                  item={item}
                  onStartQuiz={onStartQuiz}
                />
              ))}
            </div>
          )}
        </div>
      )}
    </div>
  );
}

export default ModuleCard;
