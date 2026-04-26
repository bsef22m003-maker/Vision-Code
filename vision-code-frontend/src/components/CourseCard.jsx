import { NavLink } from "react-router-dom";

function CourseCard({ course }) {
  const levelColors = {
    beginner: "bg-green-100 text-green-700",
    intermediate: "bg-purple-100 text-purple-700",
    advanced: "bg-orange-100 text-orange-700",
    "all levels": "bg-blue-100 text-blue-700",
  };

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

  if (!course) return null;

  const {
    title,
    description,
    difficulty_level,
    topics = [],
    price,
    instructor,
    total_modules,   // ✅ number from API e.g. 15
    total_students,  // ✅ number from API e.g. 3
    is_published,
  } = course;

  const formatDuration = (minutes) => {
    if (minutes < 60) return `${minutes} mins`;
    const hours = Math.floor(minutes / 60);
    const mins = minutes % 60;
    return mins > 0 ? `${hours}h ${mins}m` : `${hours}h`;
  };

  const capitalize = (str) => {
    if (!str) return "";
    return str.charAt(0).toUpperCase() + str.slice(1).toLowerCase();
  };

  const rating = 4.8;
  const reviews = "0";

  const gradient =
    levelGradients[difficulty_level?.toLowerCase()] ||
    "from-yellow-400 to-orange-500";
  const icon = levelIcons[difficulty_level?.toLowerCase()] || "📖";
  const level = capitalize(difficulty_level);
  const badge = is_published ? "PUBLISHED" : "DRAFT";
  const badgeColor = is_published ? "text-green-600" : "text-gray-600";
  const isFree = parseFloat(price) === 0;

  return (
    <div className="group relative w-full">
      <div
        className={`absolute -inset-0.5 bg-gradient-to-r ${gradient} rounded-xl opacity-0 group-hover:opacity-100 blur transition duration-500`}
      />

      <div className="relative bg-white rounded-xl shadow-md hover:shadow-2xl transition-all duration-300 overflow-hidden border border-gray-100 flex flex-col h-full">
        {/* Header */}
        <div
          className={`relative h-48 bg-gradient-to-br ${gradient} flex items-center justify-center overflow-hidden`}
        >
          <div className="absolute inset-0 bg-black/10" />
          <span className="text-6xl font-bold text-white relative z-10">
            {icon}
          </span>
          {badge && (
            <div
              className={`absolute top-3 right-3 bg-white/90 backdrop-blur-sm px-3 py-1 rounded-full text-xs font-bold ${badgeColor}`}
            >
              {badge}
            </div>
          )}
          <div className="absolute bottom-3 right-3 bg-white/90 backdrop-blur-sm px-3 py-1 rounded-full text-sm font-bold text-gray-900">
            {isFree ? "Free" : `₹${price}`}
          </div>
        </div>

        {/* Content */}
        <div className="p-6 flex-1 flex flex-col">
          <div className="flex justify-between items-start mb-3">
            <span
              className={`text-xs font-semibold px-3 py-1 rounded-full ${
                levelColors[difficulty_level?.toLowerCase()] ||
                levelColors.beginner
              }`}
            >
              {level}
            </span>
            <div className="flex items-center gap-1 text-xs text-gray-500">
              <svg className="w-4 h-4" fill="currentColor" viewBox="0 0 20 20">
                <path d="M9 4.804A7.968 7.968 0 005.5 4c-1.255 0-2.443.29-3.5.804v10A7.969 7.969 0 015.5 14c1.669 0 3.218.51 4.5 1.385A7.962 7.962 0 0114.5 14c1.255 0 2.443.29 3.5.804v-10A7.968 7.968 0 0014.5 4c-1.255 0-2.443.29-3.5.804V12a1 1 0 11-2 0V4.804z" />
              </svg>
              {total_modules} Module{total_modules !== 1 ? "s" : ""}
            </div>
          </div>

          <h3 className="text-xl font-bold text-gray-900 mb-2 group-hover:text-blue-600 transition">
            {title}
          </h3>
          <p className="text-gray-600 text-sm mb-4 flex-1 line-clamp-3">
            {description}
          </p>

          {topics.length > 0 && (
            <div className="flex flex-wrap gap-2 mb-4">
              {topics.slice(0, 3).map((topic, index) => (
                <span
                  key={index}
                  className="text-xs bg-blue-50 text-blue-700 px-2 py-1 rounded-full"
                >
                  {topic}
                </span>
              ))}
              {topics.length > 3 && (
                <span className="text-xs bg-gray-100 text-gray-600 px-2 py-1 rounded-full">
                  +{topics.length - 3} more
                </span>
              )}
            </div>
          )}

          <div className="space-y-2 mb-4">
            <div className="flex items-center gap-2 text-xs text-gray-600">
              <svg className="w-4 h-4 text-indigo-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M17 20h5v-2a3 3 0 00-5.356-1.857M17 20H7m10 0v-2c0-.656-.126-1.283-.356-1.857M7 20H2v-2a3 3 0 015.356-1.857M7 20v-2c0-.656.126-1.283.356-1.857m0 0a5.002 5.002 0 019.288 0M15 7a3 3 0 11-6 0 3 3 0 016 0z" />
              </svg>
              <span>{total_students?.toLocaleString()} students enrolled</span>
            </div>
            {instructor && (
              <div className="flex items-center gap-2 text-xs text-gray-600">
                <svg className="w-4 h-4 text-indigo-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M16 7a4 4 0 11-8 0 4 4 0 018 0zM12 14a7 7 0 00-7 7h14a7 7 0 00-7-7z" />
                </svg>
                <span className="truncate">{instructor.name}</span>
              </div>
            )}
          </div>

          <div className="flex items-center justify-between mt-auto pt-4 border-t border-gray-100">
            <div className="flex items-center gap-1">
              <div className="flex text-yellow-400">
                <svg className="w-4 h-4 fill-current" viewBox="0 0 20 20">
                  <path d="M10 15l-5.878 3.09 1.123-6.545L.489 6.91l6.572-.955L10 0l2.939 5.955 6.572.955-4.756 4.635 1.123 6.545z" />
                </svg>
              </div>
              <span className="font-bold text-gray-900">{rating}</span>
              <span className="text-gray-400 text-sm">({reviews})</span>
            </div>
            <NavLink
              to={`/course/${course.id}`}
              className="px-4 py-2 bg-gradient-to-r from-blue-600 to-purple-600 text-white rounded-lg hover:shadow-lg transition font-semibold text-sm transform hover:scale-105"
            >
              Go to Course
            </NavLink>
          </div>
        </div>
      </div>
    </div>
  );
}

export default CourseCard;