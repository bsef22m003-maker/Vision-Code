import NavBar from "../components/NavBar";
import { ArrowLeft, BookOpen, Zap, Clock, Users, Star } from "lucide-react";

export default function CourseDetailSkeleton() {
  return (
    <div className="min-h-screen bg-gray-50">
      {/* NavBar */}
      <NavBar
        appName="Vision-Code"
        authButtons={
          <div className="flex gap-3">
            <div className="px-4 py-2 h-4 bg-gray-300 rounded w-16 animate-pulse" />
            <div className="px-6 py-2 h-4 bg-blue-300 rounded w-20 animate-pulse" />
          </div>
        }
      >
        <div className="h-4 bg-gray-300 rounded w-20 animate-pulse" />
        <div className="h-4 bg-gray-300 rounded w-24 animate-pulse" />
        <div className="h-4 bg-gray-300 rounded w-20 animate-pulse" />
      </NavBar>

      {/* ── Hero Banner Skeleton ── */}
      <div className="relative bg-gradient-to-br from-blue-500 to-purple-600 overflow-hidden">
        {/* Decorative blobs */}
        <div className="absolute top-0 right-0 w-96 h-96 bg-white/10 rounded-full -translate-y-1/2 translate-x-1/2" />
        <div className="absolute bottom-0 left-0 w-64 h-64 bg-black/10 rounded-full translate-y-1/2 -translate-x-1/2" />

        <div className="relative max-w-6xl mx-auto px-6 py-12">
          {/* Back link skeleton */}
          <div className="inline-flex items-center gap-2 mb-8">
            <ArrowLeft className="w-4 h-4 text-white/50" />
            <div className="h-4 bg-white/20 rounded w-32 animate-pulse" />
          </div>

          <div className="flex flex-col lg:flex-row gap-8 items-start">
            {/* Icon skeleton */}
            <div className="w-24 h-24 bg-white/20 rounded-2xl flex-shrink-0 animate-pulse" />

            {/* Info section */}
            <div className="flex-1 space-y-4">
              {/* Badges skeleton */}
              <div className="flex flex-wrap gap-2 mb-4">
                <div className="h-6 bg-white/20 rounded-full w-24 animate-pulse" />
                <div className="h-6 bg-white/20 rounded-full w-20 animate-pulse" />
                <div className="h-6 bg-white/20 rounded-full w-28 animate-pulse" />
              </div>

              {/* Title skeleton */}
              <div className="space-y-2">
                <div className="h-8 bg-white/20 rounded w-3/4 animate-pulse" />
                <div className="h-8 bg-white/20 rounded w-2/3 animate-pulse" />
              </div>

              {/* Description skeleton */}
              <div className="space-y-2">
                <div className="h-4 bg-white/20 rounded animate-pulse" />
                <div className="h-4 bg-white/20 rounded w-5/6 animate-pulse" />
              </div>

              {/* Quick stats skeleton */}
              <div className="flex flex-wrap gap-6 text-white/90 text-sm pt-2">
                <div className="flex items-center gap-1.5">
                  <BookOpen className="w-4 h-4" />
                  <div className="h-3 bg-white/20 rounded w-16 animate-pulse" />
                </div>
                <div className="flex items-center gap-1.5">
                  <Zap className="w-4 h-4" />
                  <div className="h-3 bg-white/20 rounded w-16 animate-pulse" />
                </div>
                <div className="flex items-center gap-1.5">
                  <Clock className="w-4 h-4" />
                  <div className="h-3 bg-white/20 rounded w-20 animate-pulse" />
                </div>
                <div className="flex items-center gap-1.5">
                  <Users className="w-4 h-4" />
                  <div className="h-3 bg-white/20 rounded w-24 animate-pulse" />
                </div>
                <div className="flex items-center gap-1.5">
                  <Star className="w-4 h-4" />
                  <div className="h-3 bg-white/20 rounded w-20 animate-pulse" />
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* ── Body ─────────────────────────────────────────────────────────────── */}
      <div className="max-w-6xl mx-auto px-6 py-10">
        <div className="grid lg:grid-cols-3 gap-8">
          {/* Left — Modules */}
          <div className="lg:col-span-2 space-y-8">
            {/* Instructor card skeleton */}
            <div className="bg-white rounded-2xl border border-gray-200 p-5 flex items-center gap-4 shadow-sm animate-pulse">
              <div className="w-14 h-14 rounded-xl bg-gradient-to-br from-blue-500 to-purple-600 flex-shrink-0" />
              <div className="flex-1 space-y-2">
                <div className="h-3 bg-gray-300 rounded w-20" />
                <div className="h-4 bg-gray-300 rounded w-32" />
                <div className="h-3 bg-gray-300 rounded w-40" />
              </div>
            </div>

            {/* Topics skeleton */}
            <div className="bg-white rounded-2xl border border-gray-200 p-5 shadow-sm animate-pulse">
              <div className="h-4 bg-gray-300 rounded w-32 mb-4" />
              <div className="flex flex-wrap gap-2">
                {[1, 2, 3, 4, 5].map((idx) => (
                  <div
                    key={idx}
                    className="h-8 bg-blue-100 rounded-full w-24"
                  />
                ))}
              </div>
            </div>

            {/* Modules section skeleton */}
            <div>
              <div className="h-6 bg-gray-300 rounded w-40 mb-4 animate-pulse" />
              <div className="space-y-3">
                {[1, 2, 3, 4].map((idx) => (
                  <div
                    key={idx}
                    className="bg-white rounded-xl border border-gray-200 p-4 animate-pulse"
                  >
                    <div className="flex items-center justify-between">
                      <div className="flex items-center gap-3 flex-1">
                        <div className="w-10 h-10 bg-gray-300 rounded-lg flex-shrink-0" />
                        <div className="flex-1 space-y-2">
                          <div className="h-4 bg-gray-300 rounded w-2/3" />
                          <div className="h-3 bg-gray-300 rounded w-1/2" />
                        </div>
                      </div>
                      <div className="w-5 h-5 bg-gray-300 rounded" />
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>

          {/* Right — Enroll Card skeleton */}
          <div className="lg:sticky lg:top-8 lg:self-start">
            <div className="bg-white rounded-2xl border border-gray-200 shadow-lg overflow-hidden animate-pulse">
              {/* Card top gradient strip */}
              <div className="h-2 bg-gradient-to-r from-blue-500 to-purple-600" />

              <div className="p-6 space-y-4">
                {/* Price skeleton */}
                <div className="space-y-1">
                  <div className="h-10 bg-gray-300 rounded w-24" />
                  <div className="h-3 bg-green-100 rounded w-40" />
                </div>

                {/* Buttons skeleton */}
                <div className="h-12 bg-gradient-to-r from-blue-600 to-purple-600 rounded-xl" />
                <div className="h-12 bg-gray-100 rounded-xl" />

                {/* Divider */}
                <hr className="my-5 border-gray-100" />

                {/* Stats skeleton */}
                <div className="space-y-3">
                  {[1, 2, 3, 4, 5, 6].map((idx) => (
                    <div
                      key={idx}
                      className="flex justify-between items-center"
                    >
                      <div className="h-3 bg-gray-300 rounded w-20" />
                      <div className="h-3 bg-gray-300 rounded w-16" />
                    </div>
                  ))}
                </div>

                {/* Guarantee skeleton */}
                <div className="mt-5 p-3 bg-gray-50 rounded-lg border border-gray-100 text-center">
                  <div className="h-3 bg-gray-300 rounded w-48 mx-auto" />
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
