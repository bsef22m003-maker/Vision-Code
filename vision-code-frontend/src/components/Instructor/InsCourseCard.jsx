import { useNavigate } from "react-router-dom";

export default function InsCourseCard({ course }) {
  const navigate = useNavigate();

  return (
    <div className="group relative">
      <div
        className={`absolute -inset-0.5 bg-gradient-to-r ${course.gradient} rounded-xl opacity-0 group-hover:opacity-100 blur transition duration-500`}
      ></div>

      <div className="relative bg-white rounded-xl shadow-md hover:shadow-2xl transition-all duration-300 overflow-hidden border border-gray-100 flex flex-col">
        <div
          className={`relative h-48 bg-gradient-to-br ${course.gradient} flex items-center justify-center overflow-hidden`}
        >
          <div className="absolute inset-0 bg-black/10"></div>
          <span
            className={`${
              course.icon.length > 5 ? "text-5xl" : "text-6xl"
            } font-bold text-white relative z-10`}
          >
            {course.icon}
          </span>
          <div className="absolute top-3 right-3">
            <span
              className={`${
                course.status === "published"
                  ? "bg-green-100 text-green-700"
                  : "bg-yellow-100 text-yellow-700"
              } text-xs px-3 py-1 rounded-full font-bold uppercase backdrop-blur-sm bg-white/90`}
            >
              {course.status}
            </span>
          </div>
        </div>

        <div className="p-6 flex-1 flex flex-col">
          <h3 className="text-xl font-bold text-gray-900 mb-3 group-hover:text-blue-600 transition">
            {course.title}
          </h3>

          {/* Stats Grid */}
          <div className="grid grid-cols-2 gap-3 mb-4">
            <div className="bg-gray-50 rounded-lg p-3">
              <p className="text-xs text-gray-600 mb-1">Students</p>
              <p className="text-lg font-bold text-gray-900">
                {course.students.toLocaleString()}
              </p>
            </div>
            <div className="bg-gray-50 rounded-lg p-3">
              <p className="text-xs text-gray-600 mb-1">Revenue</p>
              <p className="text-lg font-bold text-green-600">
                ${course.revenue.toLocaleString()}
              </p>
            </div>
            <div className="bg-gray-50 rounded-lg p-3">
              <p className="text-xs text-gray-600 mb-1">Rating</p>
              <p
                className={`text-lg font-bold ${
                  course.rating ? "text-yellow-600" : "text-gray-400"
                } flex items-center gap-1`}
              >
                {course.rating || "N/A"}
                {course.rating && (
                  <span className="text-xs text-gray-500">
                    (
                    {course.reviews >= 1000
                      ? `${(course.reviews / 1000).toFixed(1)}k`
                      : course.reviews}
                    )
                  </span>
                )}
              </p>
            </div>
            <div className="bg-gray-50 rounded-lg p-3">
              <p className="text-xs text-gray-600 mb-1">Modules</p>
              <p className="text-lg font-bold text-blue-600">
                {course.modules}
              </p>
            </div>
          </div>

          <p className="text-xs text-gray-500 mb-4">
            Last updated: {course.lastUpdated}
          </p>

          {/* Actions */}
          <div className="flex gap-2 mt-auto">
            <button
              onClick={() =>
                navigate(`/instructor/courses/${course.id}/modules`)
              }
              className="flex-1 px-4 py-2 border-2 border-blue-600 text-blue-600 rounded-lg font-semibold text-sm hover:bg-blue-600 hover:text-white transition flex items-center justify-center gap-1"
            >
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
                  d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z"
                />
              </svg>
              Modules
            </button>
            <button
              onClick={() => navigate(`/instructor/courses/${course.id}/edit`)}
              className="flex-1 px-4 py-2 bg-gradient-to-r from-blue-600 to-purple-600 text-white rounded-lg font-semibold text-sm hover:shadow-lg transition flex items-center justify-center gap-1"
            >
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
                  d="M11 5H6a2 2 0 00-2 2v11a2 2 0 002 2h11a2 2 0 002-2v-5m-1.414-9.414a2 2 0 112.828 2.828L11.828 15H9v-2.828l8.586-8.586z"
                />
              </svg>
              Edit
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}
