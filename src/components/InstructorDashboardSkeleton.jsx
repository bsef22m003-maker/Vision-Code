import { Settings, LogOut, Plus } from "lucide-react";

export default function InstructorDashboardSkeleton() {
  return (
    <div className="min-h-screen bg-gray-50">
      {/* ── Navbar ── */}
      <nav className="bg-white shadow-sm border-b sticky top-0 z-50">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex justify-between items-center h-16">
            <div className="flex items-center gap-2">
              <div className="bg-gradient-to-r from-blue-600 to-purple-600 p-2 rounded-lg">
                <svg
                  xmlns="http://www.w3.org/2000/svg"
                  width="20"
                  height="20"
                  fill="none"
                  stroke="white"
                  strokeWidth="2"
                >
                  <polyline points="16 18 22 12 16 6" />
                  <polyline points="8 6 2 12 8 18" />
                </svg>
              </div>
              <span className="text-xl font-bold bg-gradient-to-r from-blue-600 to-purple-600 bg-clip-text text-transparent">
                Vision-Code
              </span>
            </div>

            <div className="flex items-center gap-3">
              <div className="flex items-center gap-2">
                <div className="w-9 h-9 bg-gray-300 rounded-full animate-pulse" />
                <div className="hidden md:block space-y-1">
                  <div className="h-3 bg-gray-300 rounded w-24 animate-pulse" />
                  <div className="h-2 bg-gray-300 rounded w-16 animate-pulse" />
                </div>
              </div>
              <button className="flex items-center gap-1.5 px-3 py-2 text-sm text-red-600">
                <LogOut className="w-4 h-4" />
                Logout
              </button>
            </div>
          </div>
        </div>
      </nav>

      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {/* ── Welcome Banner Skeleton ── */}
        <div className="bg-gradient-to-r from-blue-600 via-purple-600 to-indigo-600 rounded-2xl p-8 mb-8 relative overflow-hidden animate-pulse">
          <div className="absolute top-0 right-0 w-64 h-64 bg-white/5 rounded-full -translate-y-1/2 translate-x-1/2" />
          <div className="absolute bottom-0 left-1/2 w-48 h-48 bg-white/5 rounded-full translate-y-1/2" />
          <div className="relative z-10 flex items-center justify-between flex-wrap gap-4">
            <div className="flex-1 space-y-3">
              <div className="h-8 bg-white/20 rounded w-2/3" />
              <div className="h-4 bg-white/20 rounded w-1/2" />
            </div>
            <div className="px-4 py-2 bg-white text-blue-600 rounded-xl font-semibold text-sm flex items-center gap-2">
              <Plus className="w-4 h-4" />
              <span className="h-4 bg-gray-300 rounded w-20" />
            </div>
          </div>
        </div>

        {/* ── Stats Grid Skeleton ── */}
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-8">
          {[1, 2, 3, 4].map((idx) => (
            <div
              key={idx}
              className="bg-white rounded-xl p-5 shadow-sm border border-gray-100 animate-pulse"
            >
              <div className="flex items-center justify-between mb-3">
                <div className="h-4 bg-gray-300 rounded w-20" />
                <div className="w-9 h-9 bg-gray-300 rounded-lg" />
              </div>
              <div className="h-8 bg-gray-300 rounded w-16 mb-2" />
              <div className="h-3 bg-gray-300 rounded w-24" />
            </div>
          ))}
        </div>

        {/* ── Courses Section Skeleton ── */}
        <div className="bg-white rounded-xl shadow-sm border border-gray-100">
          <div className="flex items-center justify-between p-6 border-b border-gray-100">
            <div className="h-6 bg-gray-300 rounded w-32 animate-pulse" />
            <div className="px-4 py-2 bg-blue-600 text-white rounded-lg text-sm font-semibold flex items-center gap-2">
              <Plus className="w-4 h-4" />
              <span className="h-4 bg-blue-700 rounded w-20" />
            </div>
          </div>

          <div className="p-6">
            {/* Course Cards Skeleton */}
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-5">
              {[1, 2, 3, 4, 5, 6].map((idx) => (
                <div
                  key={idx}
                  className="bg-white rounded-xl border border-gray-200 p-5 animate-pulse"
                >
                  {/* Header with Icon and Title */}
                  <div className="flex items-start gap-4 mb-4">
                    <div className="w-12 h-12 bg-gradient-to-br from-blue-500 to-purple-600 rounded-lg flex-shrink-0" />
                    <div className="flex-1 min-w-0 space-y-2">
                      <div className="flex items-start justify-between gap-2">
                        <div className="h-4 bg-gray-300 rounded flex-1" />
                        <div className="h-5 bg-gray-300 rounded-full w-16 flex-shrink-0" />
                      </div>
                    </div>
                  </div>

                  {/* Description */}
                  <div className="space-y-2 mb-3">
                    <div className="h-3 bg-gray-300 rounded" />
                    <div className="h-3 bg-gray-300 rounded w-5/6" />
                  </div>

                  {/* Stats */}
                  <div className="flex items-center justify-between text-xs mb-4 pb-4 border-b border-gray-100">
                    <div className="flex items-center gap-4">
                      <div className="h-3 bg-gray-300 rounded w-16" />
                      <div className="h-3 bg-gray-300 rounded w-20" />
                    </div>
                  </div>

                  {/* Actions */}
                  <div className="flex gap-2">
                    <div className="flex-1 h-8 bg-blue-50 rounded-lg" />
                    <div className="w-10 h-8 border border-red-200 rounded-lg" />
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
