import { useState } from "react";
import { useParams, Link, useNavigate } from "react-router-dom";
import {
  ArrowLeft,
  BookOpen,
  HelpCircle,
  Code,
  Star,
  Clock,
  CheckCircle,
  ChevronDown,
  ChevronUp,
  Play,
  FileText,
  Trophy,
  Users,
  BarChart2,
  Layers,
  Tag,
  LogOut,
  Settings,
} from "lucide-react";

import { useAuth } from "../context/FakeAuth";
import { useCourseDetail } from "../hooks/useCourseDetail";
import { useQuery } from "@tanstack/react-query";
import { getCourseModules } from "../services/courseDetail";
import ModuleCard from "../components/ModuleCard";
import StudentCourseDetailSkeleton from "../components/StudentCourseDetailSkeleton";

// ─── Constants ────────────────────────────────────────────────────────────────

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

const getProgressColor = (progress) => {
  if (progress >= 75) return "from-green-500 to-emerald-500";
  if (progress >= 50) return "from-blue-500 to-indigo-500";
  if (progress >= 25) return "from-yellow-500 to-orange-500";
  return "from-red-500 to-pink-500";
};

export default function StudentCourseDetail() {
  const { courseId } = useParams();
  const navigate = useNavigate();
  const id = courseId ? parseInt(courseId) : 0;

  const { data: course, isLoading, error } = useCourseDetail(id);
  const { logout, user } = useAuth();

  // TODO: Replace with real enrollment/progress API data
  const progress = 0;
  const enrolled_at = undefined;
  const completed = false;

  const level = course?.difficulty_level?.toLowerCase();
  const gradient = levelGradients[level] || "from-blue-500 to-indigo-600";
  const icon = levelIcons[level] || "📖";

  const initials = user
    ? `${user.firstName?.[0] ?? ""}${user.lastName?.[0] ?? ""}`.toUpperCase()
    : "S";

  const handleLogout = () => {
    logout();
    navigate("/login");
  };

  // ── Loading ──────────────────────────────────────────────────────────────

  if (isLoading) {
    return <StudentCourseDetailSkeleton />;
  }

  // ── Error ────────────────────────────────────────────────────────────────

  if (error || !course) {
    return (
      <div className="min-h-screen bg-gray-50 flex items-center justify-center">
        <div className="text-center">
          <p className="text-xl text-gray-500 mb-4">
            {error instanceof Error ? error.message : "Course not found."}
          </p>
          <Link
            to="/student"
            className="px-5 py-2 bg-blue-600 text-white rounded-xl font-semibold text-sm hover:bg-blue-700 transition"
          >
            Back to Dashboard
          </Link>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gray-50">
      {/* ── Navbar ── */}
      <nav className="bg-white shadow-sm border-b sticky top-0 z-50">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex justify-between items-center h-16">
            <Link to="/" className="flex items-center gap-2">
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
            </Link>
            <div className="flex items-center gap-3">
              <Link
                to="/settings"
                className="p-2 text-gray-500 hover:text-blue-600 transition rounded-lg hover:bg-gray-100"
              >
                <Settings className="w-5 h-5" />
              </Link>
              <div className="w-9 h-9 bg-gradient-to-r from-blue-600 to-purple-600 rounded-full flex items-center justify-center text-white font-bold text-sm">
                {initials}
              </div>
              <button
                onClick={handleLogout}
                className="flex items-center gap-1.5 px-3 py-2 text-sm text-red-600 hover:bg-red-50 rounded-lg transition font-medium"
              >
                <LogOut className="w-4 h-4" />
                Logout
              </button>
            </div>
          </div>
        </div>
      </nav>

      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {/* ── Back Link ── */}
        <Link
          to="/student"
          className="inline-flex items-center gap-2 text-gray-500 hover:text-blue-600 text-sm font-medium mb-6 transition-colors"
        >
          <ArrowLeft className="w-4 h-4" /> Back to Dashboard
        </Link>

        {/* ── Hero Banner ── */}
        <div
          className={`bg-gradient-to-r ${gradient} rounded-2xl p-8 mb-8 text-white relative overflow-hidden`}
        >
          <div className="absolute top-0 right-0 w-64 h-64 bg-white/5 rounded-full -translate-y-1/2 translate-x-1/2" />
          <div className="absolute bottom-0 left-1/2 w-48 h-48 bg-white/5 rounded-full translate-y-1/2" />
          <div className="relative z-10 flex items-start gap-6 flex-wrap">
            <div className="w-16 h-16 bg-white/20 rounded-2xl flex items-center justify-center text-3xl flex-shrink-0 shadow-lg backdrop-blur-sm">
              {icon}
            </div>
            <div className="flex-1 min-w-0">
              <div className="flex flex-wrap items-center gap-2 mb-2">
                <span className="text-xs font-semibold bg-white/20 px-2 py-0.5 rounded-full capitalize">
                  {course.difficulty_level}
                </span>
                {completed && (
                  <span className="text-xs font-semibold bg-green-400/30 border border-green-300/50 px-2 py-0.5 rounded-full flex items-center gap-1">
                    <CheckCircle className="w-3 h-3" /> Completed
                  </span>
                )}
              </div>
              <h1 className="text-2xl sm:text-3xl font-bold mb-2">
                {course.title}
              </h1>
              <p className="text-white/80 text-sm max-w-2xl">
                {course.description}
              </p>
              <div className="flex items-center gap-4 mt-3 text-sm text-white/70">
                <span className="flex items-center gap-1.5">
                  <Users className="w-4 h-4" />
                  {course.total_students?.toLocaleString()} students
                </span>
                <span className="flex items-center gap-1.5">
                  <Layers className="w-4 h-4" />
                  {course.total_modules} modules
                </span>
                <span className="flex items-center gap-1.5">
                  <Star className="w-4 h-4 fill-white/70" />
                  4.5 rating
                </span>
              </div>
            </div>
          </div>
        </div>

        {/* ── Main Grid ── */}
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
          {/* Left — Modules */}
          <div className="lg:col-span-2 space-y-6">
            {/* Instructor */}
            <div className="bg-white rounded-xl shadow-sm border border-gray-100 p-5 flex items-center gap-4">
              <div className="w-12 h-12 bg-gradient-to-br from-blue-500 to-purple-600 rounded-xl flex items-center justify-center text-white font-bold text-lg flex-shrink-0">
                {course.instructor.name?.charAt(0).toUpperCase()}
              </div>
              <div>
                <p className="text-xs text-gray-400 font-medium">Instructor</p>
                <p className="font-bold text-gray-900">
                  {course.instructor.name}
                </p>
                <p className="text-xs text-gray-500">
                  {course.instructor.email}
                </p>
              </div>
            </div>

            {/* Modules */}
            <div>
              <div className="flex items-center justify-between mb-4">
                <h2 className="text-xl font-bold text-gray-900">
                  Course Modules
                </h2>
                <span className="text-sm text-gray-400">
                  {course.modules.length} modules
                </span>
              </div>
              <div className="space-y-3">
                {course.modules.map((module, idx) => (
                  <ModuleCard
                    key={module.id}
                    moduleData={module}
                    index={idx}
                    courseId={id}
                  />
                ))}
              </div>
            </div>
          </div>

          {/* Right — Sidebar */}
          <div className="space-y-5">
            {/* Continue CTA */}
            <div className="bg-gradient-to-br from-blue-600 via-purple-600 to-indigo-600 rounded-2xl p-6 text-white relative overflow-hidden">
              <div className="absolute top-0 right-0 w-32 h-32 bg-white/5 rounded-full -translate-y-1/2 translate-x-1/2" />
              <h3 className="font-bold text-lg mb-1 relative z-10">
                Ready to learn?
              </h3>
              <p className="text-blue-100 text-xs mb-4 relative z-10">
                Pick up where you left off and keep building your skills.
              </p>
              <button className="w-full py-2.5 bg-white text-blue-600 rounded-xl font-bold text-sm hover:bg-blue-50 transition relative z-10">
                Continue Learning →
              </button>
            </div>

            {/* Stats */}
            <div className="bg-white rounded-xl shadow-sm border border-gray-100 p-5">
              <h3 className="font-bold text-gray-900 mb-4">Course Info</h3>
              <div className="space-y-3">
                {[
                  {
                    icon: <Layers className="w-4 h-4 text-blue-500" />,
                    label: "Modules",
                    value: course.total_modules,
                    bg: "bg-blue-50",
                  },
                  {
                    icon: <Users className="w-4 h-4 text-purple-500" />,
                    label: "Students",
                    value: course.total_students?.toLocaleString(),
                    bg: "bg-purple-50",
                  },
                  {
                    icon: <Star className="w-4 h-4 text-yellow-500" />,
                    label: "Rating",
                    value: "4.5 / 5",
                    bg: "bg-yellow-50",
                  },
                  {
                    icon: <BarChart2 className="w-4 h-4 text-orange-500" />,
                    label: "Level",
                    value:
                      course.difficulty_level?.charAt(0).toUpperCase() +
                      course.difficulty_level?.slice(1),
                    bg: "bg-orange-50",
                  },
                  {
                    icon: <Trophy className="w-4 h-4 text-green-500" />,
                    label: "Status",
                    value: completed ? "Completed ✅" : "In Progress",
                    bg: "bg-green-50",
                  },
                ].map(({ icon, label, value, bg }) => (
                  <div
                    key={label}
                    className="flex items-center justify-between"
                  >
                    <div className="flex items-center gap-2 text-sm text-gray-500">
                      <div
                        className={`w-7 h-7 ${bg} rounded-lg flex items-center justify-center`}
                      >
                        {icon}
                      </div>
                      {label}
                    </div>
                    <span className="text-sm font-semibold text-gray-900">
                      {value}
                    </span>
                  </div>
                ))}
              </div>
            </div>

            {/* Topics */}
            {course.topics?.length > 0 && (
              <div className="bg-white rounded-xl shadow-sm border border-gray-100 p-5">
                <h3 className="font-bold text-gray-900 mb-3 flex items-center gap-2">
                  <Tag className="w-4 h-4 text-blue-500" />
                  Topics
                </h3>
                <div className="flex flex-wrap gap-2">
                  {course.topics.slice(0, 8).map((topic) => (
                    <span
                      key={topic}
                      className="text-xs px-2.5 py-1 bg-blue-50 text-blue-700 border border-blue-100 rounded-full font-medium"
                    >
                      {topic}
                    </span>
                  ))}
                  {course.topics.length > 8 && (
                    <span className="text-xs px-2.5 py-1 text-gray-400 font-medium">
                      +{course.topics.length - 8} more
                    </span>
                  )}
                </div>
              </div>
            )}

            {/* Daily Challenge */}
            <div className="bg-gradient-to-br from-orange-500 to-pink-600 rounded-xl p-5 text-white">
              <div className="flex items-center justify-between mb-2">
                <h3 className="font-bold">Daily Challenge</h3>
                <span className="text-2xl">🎯</span>
              </div>
              <p className="text-sm text-orange-50 mb-4">
                Complete today's challenge and earn 50 XP bonus!
              </p>
              <button className="w-full py-2 bg-white/20 hover:bg-white/30 border border-white/30 text-white rounded-lg text-sm font-semibold transition">
                Start Challenge
              </button>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
