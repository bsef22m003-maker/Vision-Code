import { Settings, LogOut } from "lucide-react";

export default function StudentCourseDetailSkeleton() {
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
              <Settings className="w-5 h-5 text-gray-500" />
              <div className="w-9 h-9 bg-gray-300 rounded-full" />
              <button className="flex items-center gap-1.5 px-3 py-2 text-sm text-red-600 font-medium">
                <LogOut className="w-4 h-4" />
                Logout
              </button>
            </div>
          </div>
        </div>
      </nav>

      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {/* ── Back Link ── */}
        <div className="h-4 bg-gray-300 rounded w-32 mb-6 animate-pulse" />

        {/* ── Hero Banner Skeleton ── */}
        <div className="bg-gradient-to-r from-blue-500 to-purple-600 rounded-2xl p-8 mb-8 relative overflow-hidden">
          <div className="absolute inset-0 bg-white/5" />
          <div className="relative z-10 flex items-start gap-6 flex-wrap">
            {/* Icon Skeleton */}
            <div className="w-16 h-16 bg-white/20 rounded-2xl flex-shrink-0 animate-pulse" />

            {/* Content Skeleton */}
            <div className="flex-1 min-w-0 space-y-3">
              <div className="h-3 bg-white/20 rounded w-24 animate-pulse" />
              <div className="h-8 bg-white/20 rounded w-3/4 animate-pulse" />
              <div className="space-y-2">
                <div className="h-3 bg-white/20 rounded animate-pulse" />
                <div className="h-3 bg-white/20 rounded w-5/6 animate-pulse" />
              </div>
              <div className="flex items-center gap-4 pt-3">
                <div className="h-3 bg-white/20 rounded w-32 animate-pulse" />
                <div className="h-3 bg-white/20 rounded w-24 animate-pulse" />
              </div>
            </div>
          </div>
        </div>

        {/* ── Main Grid ── */}
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
          {/* Left — Modules */}
          <div className="lg:col-span-2 space-y-6">
            {/* Instructor Skeleton */}
            <div className="bg-white rounded-xl shadow-sm border border-gray-100 p-5 flex items-center gap-4 animate-pulse">
              <div className="w-12 h-12 bg-gray-300 rounded-xl flex-shrink-0" />
              <div className="flex-1 space-y-2">
                <div className="h-3 bg-gray-300 rounded w-20" />
                <div className="h-4 bg-gray-300 rounded w-32" />
                <div className="h-3 bg-gray-300 rounded w-40" />
              </div>
            </div>

            {/* Modules Header Skeleton */}
            <div>
              <div className="h-6 bg-gray-300 rounded w-40 mb-4 animate-pulse" />

              {/* Module Cards Skeleton */}
              <div className="space-y-3">
                {[1, 2, 3, 4].map((idx) => (
                  <div
                    key={idx}
                    className="bg-white rounded-xl shadow-sm border border-gray-100 p-5 animate-pulse"
                  >
                    <div className="flex items-center gap-4">
                      <div className="w-10 h-10 bg-gray-300 rounded-xl flex-shrink-0" />
                      <div className="flex-1 space-y-2">
                        <div className="h-4 bg-gray-300 rounded w-2/3" />
                        <div className="h-3 bg-gray-300 rounded w-1/2" />
                      </div>
                      <div className="hidden sm:flex gap-3">
                        <div className="h-3 bg-gray-300 rounded w-20" />
                        <div className="h-3 bg-gray-300 rounded w-16" />
                      </div>
                      <div className="w-4 h-4 bg-gray-300 rounded" />
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>

          {/* Right — Sidebar */}
          <div className="space-y-5">
            {/* Continue CTA Skeleton */}
            <div className="bg-gradient-to-br from-blue-600 via-purple-600 to-indigo-600 rounded-2xl p-6 relative overflow-hidden animate-pulse">
              <div className="space-y-3">
                <div className="h-5 bg-white/20 rounded w-32" />
                <div className="h-3 bg-white/20 rounded w-full" />
                <div className="h-3 bg-white/20 rounded w-3/4" />
                <div className="h-10 bg-white/20 rounded-xl mt-4" />
              </div>
            </div>

            {/* Stats Skeleton */}
            <div className="bg-white rounded-xl shadow-sm border border-gray-100 p-5 animate-pulse">
              <div className="h-5 bg-gray-300 rounded w-24 mb-4" />
              <div className="space-y-3">
                {[1, 2, 3, 4, 5].map((idx) => (
                  <div key={idx} className="flex items-center justify-between">
                    <div className="flex items-center gap-2">
                      <div className="w-7 h-7 bg-gray-300 rounded-lg" />
                      <div className="h-3 bg-gray-300 rounded w-16" />
                    </div>
                    <div className="h-3 bg-gray-300 rounded w-20" />
                  </div>
                ))}
              </div>
            </div>

            {/* Topics Skeleton */}
            <div className="bg-white rounded-xl shadow-sm border border-gray-100 p-5 animate-pulse">
              <div className="h-5 bg-gray-300 rounded w-16 mb-3" />
              <div className="flex flex-wrap gap-2">
                {[1, 2, 3, 4].map((idx) => (
                  <div
                    key={idx}
                    className="h-6 bg-gray-300 rounded-full w-20"
                  />
                ))}
              </div>
            </div>

            {/* Daily Challenge Skeleton */}
            <div className="bg-gradient-to-br from-orange-500 to-pink-600 rounded-xl p-5 animate-pulse">
              <div className="space-y-3">
                <div className="h-5 bg-white/20 rounded w-32" />
                <div className="h-3 bg-white/20 rounded w-full" />
                <div className="h-3 bg-white/20 rounded w-3/4" />
                <div className="h-9 bg-white/20 rounded-lg mt-2" />
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
