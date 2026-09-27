export default function CourseGridSkeleton() {
  return (
    <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 pb-20">
      {/* Header */}
      <div className="flex items-center justify-between mb-8">
        <div className="h-8 bg-gray-300 rounded w-64 animate-pulse" />

        <div className="flex gap-2">
          <div className="p-2 rounded-lg w-16 h-10 bg-blue-600 animate-pulse" />
          <div className="p-2 rounded-lg w-16 h-10 bg-gray-200 animate-pulse" />
        </div>
      </div>

      {/* Grid of Skeleton Course Cards */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-8">
        {[1, 2, 3, 4, 5, 6].map((n) => (
          <div
            key={n}
            className="relative bg-white rounded-xl shadow-md border border-gray-100 overflow-hidden flex flex-col h-full animate-pulse"
          >
            {/* Header Gradient Skeleton */}
            <div className="relative h-48 bg-gradient-to-br from-blue-400 to-purple-500 overflow-hidden" />

            {/* Content */}
            <div className="p-6 flex-1 flex flex-col space-y-4">
              {/* Difficulty Badge and Modules */}
              <div className="flex justify-between items-start">
                <div className="h-6 bg-gray-300 rounded-full w-20" />
                <div className="h-4 bg-gray-300 rounded w-24" />
              </div>

              {/* Title */}
              <div className="space-y-2">
                <div className="h-6 bg-gray-300 rounded w-4/5" />
                <div className="h-6 bg-gray-300 rounded w-3/4" />
              </div>

              {/* Description */}
              <div className="space-y-2">
                <div className="h-4 bg-gray-300 rounded" />
                <div className="h-4 bg-gray-300 rounded" />
                <div className="h-4 bg-gray-300 rounded w-5/6" />
              </div>

              {/* Topics */}
              <div className="flex flex-wrap gap-2">
                <div className="h-6 bg-blue-100 rounded-full w-20" />
                <div className="h-6 bg-blue-100 rounded-full w-24" />
                <div className="h-6 bg-blue-100 rounded-full w-16" />
              </div>

              {/* Stats - Students and Instructor */}
              <div className="space-y-2 pt-2">
                <div className="h-4 bg-gray-300 rounded w-3/4" />
                <div className="h-4 bg-gray-300 rounded w-2/3" />
              </div>
            </div>
          </div>
        ))}
      </div>

      {/* Load More Button Skeleton */}
      <div className="text-center mt-12">
        <div className="inline-block h-12 bg-gradient-to-r from-blue-600 to-purple-600 rounded-lg w-40 animate-pulse" />
      </div>
    </div>
  );
}
