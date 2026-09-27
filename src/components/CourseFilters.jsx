import { useState } from "react";

function CourseFilters() {
  const [filters, setFilters] = useState({
    level: "All Levels",
    duration: "All Durations",
    language: "All Languages",
    sort: "Most Popular",
  });

  const handleFilterChange = (filterType, value) => {
    setFilters({ ...filters, [filterType]: value });
  };

  return (
    <div className="max-w-7xl mx-auto px-4 sm:px-6 lg: px-8 mb-8">
      <div className="bg-white rounded-xl shadow-sm border border-gray-100 p-4">
        <div className="flex flex-wrap items-center justify-between gap-4">
          <div className="flex items-center gap-4 flex-wrap">
            <span className="text-sm font-semibold text-gray-700">
              Filter by:
            </span>

            <select
              value={filters.level}
              onChange={(e) => handleFilterChange("level", e.target.value)}
              className="px-4 py-2 border border-gray-300 rounded-lg text-sm focus:ring-2 focus:ring-blue-500 focus:border-blue-500 outline-none"
            >
              <option>All Levels</option>
              <option>Beginner</option>
              <option>Intermediate</option>
              <option>Advanced</option>
            </select>

            <select
              value={filters.duration}
              onChange={(e) => handleFilterChange("duration", e.target.value)}
              className="px-4 py-2 border border-gray-300 rounded-lg text-sm focus:ring-2 focus:ring-blue-500 focus:border-blue-500 outline-none"
            >
              <option>All Durations</option>
              <option>0-5 hours</option>
              <option>5-10 hours</option>
              <option>10+ hours</option>
            </select>

            <select
              value={filters.language}
              onChange={(e) => handleFilterChange("language", e.target.value)}
              className="px-4 py-2 border border-gray-300 rounded-lg text-sm focus:ring-2 focus:ring-blue-500 focus:border-blue-500 outline-none"
            >
              <option>All Languages</option>
              <option>JavaScript</option>
              <option>Python</option>
              <option>Java</option>
              <option>C++</option>
            </select>
          </div>

          <div className="flex items-center gap-2">
            <span className="text-sm font-semibold text-gray-700">Sort: </span>
            <select
              value={filters.sort}
              onChange={(e) => handleFilterChange("sort", e.target.value)}
              className="px-4 py-2 border border-gray-300 rounded-lg text-sm focus:ring-2 focus:ring-blue-500 focus:border-blue-500 outline-none"
            >
              <option>Most Popular</option>
              <option>Highest Rated</option>
              <option>Newest</option>
              <option>Price: Low to High</option>
            </select>
          </div>
        </div>
      </div>
    </div>
  );
}

export default CourseFilters;
