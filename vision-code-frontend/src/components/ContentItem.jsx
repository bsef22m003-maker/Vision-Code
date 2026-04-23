import { useState } from "react";
import ArticleContent from "./ArticleContent";

function ContentItem({ item, onStartQuiz }) {
  const [showContent, setShowContent] = useState(false);

  const contentTypeIcons = {
    article: (
      <svg
        className="w-5 h-5"
        fill="none"
        stroke="currentColor"
        viewBox="0 0 24 24"
      >
        <path
          strokeLinecap="round"
          strokeLinejoin="round"
          strokeWidth="2"
          d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z"
        />
      </svg>
    ),
    practice_material: (
      <svg
        className="w-5 h-5"
        fill="none"
        stroke="currentColor"
        viewBox="0 0 24 24"
      >
        <path
          strokeLinecap="round"
          strokeLinejoin="round"
          strokeWidth="2"
          d="M10 20l4-16m4 4l4 4-4 4M6 16l-4-4 4-4"
        />
      </svg>
    ),
    quiz: (
      <svg
        className="w-5 h-5"
        fill="none"
        stroke="currentColor"
        viewBox="0 0 24 24"
      >
        <path
          strokeLinecap="round"
          strokeLinejoin="round"
          strokeWidth="2"
          d="M9 5H7a2 2 0 00-2 2v12a2 2 0 002 2h10a2 2 0 002-2V7a2 2 0 00-2-2h-2M9 5a2 2 0 002 2h2a2 2 0 002-2M9 5a2 2 0 012-2h2a2 2 0 012 2m-6 9l2 2 4-4"
        />
      </svg>
    ),
  };

  const contentTypeColors = {
    article: "bg-blue-50 text-blue-600 border-blue-200",
    practice_material: "bg-orange-50 text-orange-600 border-orange-200",
    quiz: "bg-green-50 text-green-600 border-green-200",
  };

  const contentTypeLabels = {
    article: "Article",
    practice_material: "Practice Material",
    quiz: "Quiz",
  };

  const difficultyColors = {
    easy: "bg-green-100 text-green-700",
    medium: "bg-yellow-100 text-yellow-700",
    hard: "bg-red-100 text-red-700",
  };

  const handleClick = () => {
    if (item.content_type === "quiz") {
      try {
        const quizData = JSON.parse(item.content);
        onStartQuiz({ ...item, parsedContent: quizData });
      } catch (error) {
        console.error("Error parsing quiz data:", error);
      }
    } else {
      // Toggle content visibility for articles and practice materials
      setShowContent(!showContent);
    }
  };

  return (
    <div
      className={`rounded-lg border-2 ${
        contentTypeColors[item.content_type]
      } overflow-hidden`}
    >
      {/* Content Header */}
      <div
        onClick={handleClick}
        className="p-4 cursor-pointer hover:shadow-md transition-all duration-300 group"
      >
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-3 flex-1">
            <div className="p-2 bg-white rounded-lg group-hover:scale-110 transition-transform">
              {contentTypeIcons[item.content_type]}
            </div>
            <div className="flex-1">
              <div className="flex items-center gap-2 mb-1">
                <span
                  className={`text-xs px-2 py-0.5 rounded-full font-semibold ${
                    contentTypeColors[item.content_type]
                  }`}
                >
                  {contentTypeLabels[item.content_type]}
                </span>
              </div>
              <h5 className="font-semibold text-gray-900 group-hover:text-blue-600 transition">
                {item.title}
              </h5>
              <div className="flex items-center gap-3 mt-1">
                <span className="text-xs text-gray-600 flex items-center gap-1">
                  <svg
                    className="w-4 h-4"
                    fill="none"
                    stroke="currentColor"
                    viewBox="0 0 24 24"
                  >
                    <path
                      strokeLinecap="round"
                      strokeLinejoin="round"
                      strokeWidth="2"
                      d="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z"
                    />
                  </svg>
                  {item.estimated_duration_minutes} min
                </span>
                <span
                  className={`text-xs px-2 py-0.5 rounded-full ${
                    difficultyColors[item.difficulty]
                  }`}
                >
                  {item.difficulty}
                </span>
                {item.content_type === "article" &&
                  item.metadata?.reading_time && (
                    <span className="text-xs text-gray-600">
                      📖 {item.metadata.reading_time}
                    </span>
                  )}
                {item.content_type === "quiz" &&
                  item.metadata?.question_count && (
                    <span className="text-xs text-gray-600">
                      ❓ {item.metadata.question_count} questions
                    </span>
                  )}
              </div>
            </div>
          </div>

          {item.content_type === "quiz" ? (
            <button className="px-4 py-2 bg-gradient-to-r from-green-600 to-emerald-600 text-white rounded-lg font-semibold text-sm hover:shadow-lg transition-all duration-300 transform group-hover:scale-105">
              Start Quiz
            </button>
          ) : (
            <svg
              className={`w-6 h-6 text-gray-400 transition-transform duration-300 ${
                showContent ? "rotate-180" : ""
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
          )}
        </div>
      </div>

      {/* Expanded Content for Articles and Practice Materials */}
      {showContent && item.content_type !== "quiz" && (
        <div className="border-t-2 border-current p-6 bg-white">
          {item.content_type === "article" && (
            <ArticleContent content={item.content} metadata={item.metadata} />
          )}
        </div>
      )}
    </div>
  );
}

export default ContentItem;
