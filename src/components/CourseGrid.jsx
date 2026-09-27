import CourseCard from "./CourseCard";
import Button from "./Button";
import CourseGridSkeleton from "./CourseGridSkeleton";
import { useQuery } from "@tanstack/react-query";
import { callCoursesApi } from "../services/coursesApi";
import { useState } from "react";

function CourseGrid() {
  const [viewMode, setViewMode] = useState("grid");

  const { data, isLoading, error } = useQuery({
    queryKey: ["courses"],
    queryFn: callCoursesApi,
  });
  const courses = Array.isArray(data) ? data : (data?.results ?? []);

  return (
    <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 pb-20">
      {/* Header */}
      <div className="flex items-center justify-between mb-8">
        <h2 className="text-2xl font-bold text-gray-900">
          Available Courses{" "}
          <span className="text-blue-600">({courses.length})</span>
        </h2>

        <div className="flex gap-2">
          {["grid", "list"].map((mode) => (
            <button
              key={mode}
              onClick={() => setViewMode(mode)}
              className={`p-2 rounded-lg capitalize ${
                viewMode === mode
                  ? "bg-blue-600 text-white"
                  : "bg-gray-200 text-gray-700 hover:bg-gray-300"
              }`}
            >
              {mode}
            </button>
          ))}
        </div>
      </div>

      {/* Loading */}
      {isLoading && <CourseGridSkeleton />}

      {/* Error */}
      {error && !isLoading && (
        <p className="text-red-500">
          {error instanceof Error ? error.message : "Something went wrong"}
        </p>
      )}

      {/* Empty */}
      {!isLoading && !error && courses.length === 0 && (
        <p className="text-gray-500 text-center py-12">No courses available</p>
      )}

      {/* Courses */}
      {!isLoading && !error && courses.length > 0 && (
        <div
          className={
            viewMode === "grid"
              ? "grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-8"
              : "flex flex-col gap-6"
          }
        >
          {courses.map((course) => (
            <CourseCard key={course.id} course={course} />
          ))}
        </div>
      )}
    </div>
  );
}

export default CourseGrid;
