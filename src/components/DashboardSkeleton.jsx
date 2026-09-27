import {
  ArrowRight,
  BookOpen,
  Clock,
  CheckCircle,
  TrendingUp,
  Settings,
  LogOut,
} from "lucide-react";

export default function DashboardSkeleton() {
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
                  <div className="h-2 bg-gray-300 rounded w-32 animate-pulse" />
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
              <div className="h-8 bg-white/20 rounded w-3/4" />
              <div className="h-4 bg-white/20 rounded w-1/2" />
            </div>
            <div className="px-4 py-2 bg-white text-blue-600 rounded-xl font-semibold text-sm">
              Browse Courses
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

        {/* ── Main Grid ── */}
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
          {/* Left — Courses */}
          <div className="lg:col-span-2">
            <div className="bg-white rounded-xl shadow-sm border border-gray-100 animate-pulse">
              {/* Header */}
              <div className="flex items-center justify-between p-6 pb-0">
                <div className="h-6 bg-gray-300 rounded w-32" />
                <div className="flex items-center gap-1 h-4 bg-gray-300 rounded w-24" />
              </div>

              {/* Tabs */}
              <div className="flex gap-4 px-6 mt-4 border-b border-gray-100 pb-3">
                <div className="h-4 bg-gray-300 rounded w-40" />
                <div className="h-4 bg-gray-300 rounded w-40" />
              </div>

              <div className="p-6 space-y-3">
                {[1, 2, 3, 4].map((idx) => (
                  <div
                    key={idx}
                    className="border border-gray-200 rounded-xl p-4 animate-pulse"
                  >
                    <div className="flex items-start gap-4">
                      <div className="w-14 h-14 bg-gray-300 rounded-xl flex-shrink-0" />
                      <div className="flex-1 min-w-0 space-y-2">
                        <div className="flex items-start justify-between gap-2">
                          <div className="h-4 bg-gray-300 rounded flex-1" />
                          <div className="h-5 bg-gray-300 rounded-full w-20 flex-shrink-0" />
                        </div>
                        <div className="h-3 bg-gray-300 rounded w-3/4" />
                        <div className="w-full bg-gray-100 rounded-full h-1.5" />
                        <div className="h-3 bg-gray-300 rounded w-1/2" />
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>

          {/* Right — Sidebar */}
          <div className="space-y-6">
            {/* Profile Card Skeleton */}
            <div className="bg-white rounded-xl shadow-sm border border-gray-100 p-6 animate-pulse">
              <div className="flex items-center gap-3 mb-4">
                <div className="w-14 h-14 bg-gradient-to-br from-blue-500 to-purple-600 rounded-xl flex-shrink-0" />
                <div className="flex-1 space-y-1">
                  <div className="h-4 bg-gray-300 rounded w-24" />
                  <div className="h-3 bg-gray-300 rounded w-32" />
                  <div className="h-5 bg-blue-100 rounded-full w-20 mt-1" />
                </div>
              </div>
              <div className="grid grid-cols-2 gap-3 text-center">
                <div className="bg-gray-50 rounded-lg p-3">
                  <div className="h-6 bg-gray-300 rounded w-8 mx-auto mb-1" />
                  <div className="h-3 bg-gray-300 rounded w-16 mx-auto" />
                </div>
                <div className="bg-gray-50 rounded-lg p-3">
                  <div className="h-6 bg-gray-300 rounded w-8 mx-auto mb-1" />
                  <div className="h-3 bg-gray-300 rounded w-20 mx-auto" />
                </div>
              </div>
            </div>

            {/* Daily Challenge Skeleton */}
            {/* <div className="bg-gradient-to-br from-orange-500 to-pink-600 rounded-xl p-6 animate-pulse">
              <div className="flex items-center justify-between mb-3">
                <div className="h-5 bg-white/20 rounded w-32" />
                <span className="text-2xl">🎯</span>
              </div>
              <div className="h-4 bg-white/20 rounded w-3/4 mb-4" />
              <div className="w-full h-10 bg-white/20 rounded-lg" />
            </div> */}

            {/* Achievements Skeleton */}
            <div className="bg-white rounded-xl shadow-sm border border-gray-100 p-6 animate-pulse">
              <div className="h-5 bg-gray-300 rounded w-32 mb-4" />
              <div className="space-y-3">
                {[1, 2, 3].map((idx) => (
                  <div key={idx} className="flex items-center gap-3">
                    <div className="w-12 h-12 bg-gray-300 rounded-lg flex-shrink-0" />
                    <div className="flex-1 space-y-1">
                      <div className="h-3 bg-gray-300 rounded w-24" />
                      <div className="h-2 bg-gray-300 rounded w-32" />
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
