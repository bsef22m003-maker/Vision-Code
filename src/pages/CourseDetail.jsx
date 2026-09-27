import { useParams, Link, useNavigate, NavLink } from "react-router-dom";
import { toast } from "sonner";
import NavBar from "../components/NavBar";
import { useAuth } from "../context/FakeAuth";
import CourseDetailSkeleton from "../components/CourseDetailSkeleton";

import { useCourseDetail } from "../hooks/useCourseDetail";
import { enrollCourse } from "../services/enrollmentApi";
import { use, useState } from "react";
import { APP_NAME } from "./HomePage";
import {
  ArrowLeft,
  BookOpen,
  HelpCircle,
  Code,
  Star,
  Clock,
  Users,
  Trophy,
  PlayCircle,
  CheckCircle,
  ChevronDown,
  ChevronUp,
  Zap,
} from "lucide-react";
import ModuleAccordion from "../components/ModuleAccridon";
import { capitalize, formatDuration } from "../utils/util";

// ─── Constants (mirrors CourseCard) ──────────────────────────────────────────

const levelGradients = {
  beginner: "from-green-400 to-emerald-500",
  intermediate: "from-purple-500 to-indigo-600",
  advanced: "from-orange-400 to-red-500",
  "all levels": "from-blue-500 to-cyan-600",
};

const levelColors = {
  beginner: "bg-green-100 text-green-700 border-green-200",
  intermediate: "bg-purple-100 text-purple-700 border-purple-200",
  advanced: "bg-orange-100 text-orange-700 border-orange-200",
  "all levels": "bg-blue-100 text-blue-700 border-blue-200",
};

const levelIcons = {
  beginner: "📚",
  intermediate: "🚀",
  advanced: "⚡",
  "all levels": "🎯",
};

// ─── Module Accordion ─────────────────────────────────────────────────────────

// ─── Main Component ───────────────────────────────────────────────────────────

export default function CourseDetail() {
  const { id } = useParams();
  const navigate = useNavigate();
  const courseId = id ? parseInt(id) : 0;

  const { data: course, isLoading, error } = useCourseDetail(courseId);
  const [isEnrolling, setIsEnrolling] = useState(false);
  const [enrollError, setEnrollError] = useState("");

  const token = localStorage.getItem("access_token");

  const handleEnrollClick = async () => {
    if (!token) {
      navigate("/login");
      return;
    }
    setIsEnrolling(true);
    setEnrollError("");
    try {
      const token = localStorage.getItem("access_token");

      if (!token) {
        setEnrollError("Authentication token not found. Please login again.");
        return;
      }
      await enrollCourse(courseId, token);
      toast(`Successfully enrolled in ${course?.title}!`);
      navigate(`/app`);
    } catch (err) {
      setEnrollError(
        err instanceof Error
          ? err.message
          : "Enrollment failed. Please try again.",
      );
    } finally {
      setIsEnrolling(false);
    }
  };

  // ── Loading ────────────────────────────────────────────────────────────────
  if (isLoading) {
    return <CourseDetailSkeleton />;
  }

  // ── Error ──────────────────────────────────────────────────────────────────
  if (error || !course) {
    return (
      <div className="min-h-screen bg-gray-50 flex items-center justify-center">
        <div className="text-center">
          <p className="text-2xl font-bold text-gray-800 mb-2">
            Course not found
          </p>
          <p className="text-gray-500 mb-6">
            {error instanceof Error ? error.message : "Something went wrong."}
          </p>
          <Link
            to="/courses"
            className="px-6 py-3 bg-blue-600 text-white rounded-xl font-semibold hover:bg-blue-700 transition"
          >
            Back to Courses
          </Link>
        </div>
      </div>
    );
  }

  // ── Derived values ─────────────────────────────────────────────────────────
  const level = course.difficulty_level?.toLowerCase();
  const gradient = levelGradients[level] || "from-yellow-400 to-orange-500";
  const levelColor = levelColors[level] || levelColors.beginner;
  const icon = levelIcons[level] || "📖";
  const isFree = parseFloat(course.price) === 0;
  const badge = course.is_published ? "PUBLISHED" : "DRAFT";
  const badgeColor = course.is_published
    ? "bg-green-100 text-green-700"
    : "bg-gray-100 text-gray-600";

  const totalDuration = course.modules?.reduce((sum, mod) => {
    return (
      sum +
      (mod.content_items?.reduce(
        (s, item) => s + (item.estimated_duration_minutes || 0),
        0,
      ) || 0)
    );
  }, 0);

  const totalItems = course.modules?.reduce(
    (sum, mod) => sum + (mod.content_items?.length || 0),
    0,
  );

  return (
    <div className="min-h-screen bg-gray-50">
      <NavBar
        appName={APP_NAME}
        authButtons={
          <>
            <NavLink
              to="/login"
              className="px-4 py-2 text-blue-600 hover:text-blue-700 font-medium transition"
            >
              Login
            </NavLink>
            <NavLink
              to="/signup"
              className="px-6 py-2 bg-gradient-to-r from-blue-600 to-purple-600 text-white rounded-lg hover:shadow-lg transition"
            >
              Sign Up
            </NavLink>
          </>
        }
      >
        <NavLink
          to="/courses"
          className={({ isActive }) =>
            `font-medium transition pb-1 ${
              isActive
                ? "text-blue-600 border-b-2 border-blue-600"
                : "text-gray-700 hover:text-blue-600"
            }`
          }
        >
          Courses
        </NavLink>
      </NavBar>
      {/* ── Hero Banner (mirrors CourseCard header) ── */}
      <div className={`relative bg-gradient-to-br ${gradient} overflow-hidden`}>
        {/* Decorative blobs */}
        <div className="absolute top-0 right-0 w-96 h-96 bg-white/10 rounded-full -translate-y-1/2 translate-x-1/2" />
        <div className="absolute bottom-0 left-0 w-64 h-64 bg-black/10 rounded-full translate-y-1/2 -translate-x-1/2" />

        <div className="relative max-w-6xl mx-auto px-6 py-12">
          {/* Back link */}
          <Link
            to="/courses"
            className="inline-flex items-center gap-2 text-white/80 hover:text-white text-sm font-medium mb-8 transition-colors"
          >
            <ArrowLeft className="w-4 h-4" /> Back to Courses
          </Link>

          <div className="flex flex-col lg:flex-row gap-8 items-start">
            {/* Icon */}
            <div className="w-24 h-24 bg-white/20 backdrop-blur-sm rounded-2xl flex items-center justify-center text-5xl flex-shrink-0 shadow-lg">
              {icon}
            </div>

            {/* Info */}
            <div className="flex-1">
              {/* Badges row */}
              <div className="flex flex-wrap gap-2 mb-4">
                <span
                  className={`text-xs font-bold px-3 py-1 rounded-full border ${levelColor}`}
                >
                  {capitalize(course.difficulty_level)}
                </span>
                <span
                  className={`text-xs font-bold px-3 py-1 rounded-full ${badgeColor}`}
                >
                  {badge}
                </span>
                {course.topics?.slice(0, 2).map((topic) => (
                  <span
                    key={topic}
                    className="text-xs px-3 py-1 rounded-full bg-white/20 text-white font-medium"
                  >
                    {topic}
                  </span>
                ))}
              </div>

              <h1 className="text-3xl lg:text-4xl font-extrabold text-white mb-3 leading-tight">
                {course.title}
              </h1>
              <p className="text-white/80 text-base mb-6 max-w-2xl leading-relaxed">
                {course.description}
              </p>

              {/* Quick stats row */}
              <div className="flex flex-wrap gap-6 text-white/90 text-sm">
                <span className="flex items-center gap-1.5">
                  <BookOpen className="w-4 h-4" />
                  {course.total_modules} modules
                </span>
                {/* <span className="flex items-center gap-1.5">
                  <Zap className="w-4 h-4" />
                  {totalItems} lessons
                </span>
                <span className="flex items-center gap-1.5">
                  <Clock className="w-4 h-4" />
                  {formatDuration(totalDuration)}
                </span> */}
                <span className="flex items-center gap-1.5">
                  <Users className="w-4 h-4" />
                  {course.total_students?.toLocaleString()} students
                </span>
                <span className="flex items-center gap-1.5">
                  <Star className="w-4 h-4 fill-yellow-300 text-yellow-300" />
                  4.5 rating
                </span>
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
            {/* Instructor card */}
            <div className="bg-white rounded-2xl border border-gray-200 p-5 flex items-center gap-4 shadow-sm">
              <div
                className={`w-14 h-14 rounded-xl bg-gradient-to-br ${gradient} flex items-center justify-center text-white font-extrabold text-xl shadow`}
              >
                {course.instructor?.name?.charAt(0).toUpperCase()}
              </div>
              <div>
                <p className="text-xs text-gray-400 uppercase tracking-wider font-semibold mb-0.5">
                  Instructor
                </p>
                <p className="font-bold text-gray-900">
                  {course.instructor?.name}
                </p>
                <p className="text-sm text-gray-500">
                  {course.instructor?.email}
                </p>
              </div>
            </div>

            {/* Topics ]

            {course.topics?.length > 0 && (
              <div className="bg-white rounded-2xl border border-gray-200 p-5 shadow-sm">
                <p className="text-xs font-semibold text-gray-500 uppercase tracking-wider mb-3">
                  Topics Covered
                </p>
                <div className="flex flex-wrap gap-2">
                  {course.topics.map((topic) => (
                    <span
                      key={topic}
                      className="text-sm bg-blue-50 text-blue-700 border border-blue-100 px-3 py-1.5 rounded-full font-medium"
                    >
                      {topic}
                    </span>
                  ))}
                </div>
              </div>
            )}

            {/* Modules */}
            <div>
              <h2 className="text-xl font-extrabold text-gray-900 mb-4 flex items-center gap-2">
                <span
                  className={`w-1.5 h-6 rounded-full bg-gradient-to-b ${gradient} inline-block`}
                />
                Course Modules
              </h2>
              <div className="space-y-3">
                {course.modules?.map((module, idx) => (
                  <ModuleAccordion
                    key={module.id}
                    moduleData={module}
                    idx={idx}
                  />
                ))}
              </div>
            </div>
          </div>

          {/* Right — Sticky Enroll Card */}
          <div className="lg:sticky lg:top-8 lg:self-start">
            <div className="bg-white rounded-2xl border border-gray-200 shadow-lg overflow-hidden">
              {/* Card top gradient strip */}
              <div className={`h-2 bg-gradient-to-r ${gradient}`} />

              <div className="p-6">
                {/* Price */}
                <div className="flex items-end gap-2 mb-1">
                  <span className="text-4xl font-extrabold text-gray-900">
                    {isFree ? "Free" : `₹${course.price}`}
                  </span>
                  {!isFree && (
                    <span className="text-sm text-gray-400 line-through mb-1">
                      ₹{(parseFloat(course.price) * 1.3).toFixed(0)}
                    </span>
                  )}
                </div>
                {!isFree && (
                  <p className="text-xs text-green-600 font-semibold mb-4">
                    🎉 23% off — Limited time!
                  </p>
                )}

                {/* Error */}
                {enrollError && (
                  <div className="mb-4 p-3 bg-red-50 border border-red-200 rounded-lg text-sm text-red-700">
                    {enrollError}
                  </div>
                )}

                {/* Auth notice */}
                {!token && (
                  <div className="mb-4 p-3 bg-blue-50 border border-blue-100 rounded-lg text-xs text-blue-700">
                    <p className="font-semibold mb-0.5">Sign in required</p>
                    <p>Please sign in to enroll in this course.</p>
                  </div>
                )}

                {/* Enroll button */}
                <button
                  onClick={handleEnrollClick}
                  disabled={isEnrolling}
                  className={`w-full py-3.5 rounded-xl font-bold text-sm text-white bg-gradient-to-r ${gradient} hover:opacity-90 transition-all duration-200 shadow-md hover:shadow-lg transform hover:-translate-y-0.5 disabled:opacity-50 disabled:cursor-not-allowed disabled:transform-none mb-3`}
                >
                  {isEnrolling
                    ? "Enrolling..."
                    : token
                      ? isFree
                        ? "Enroll Free"
                        : "Enroll Now"
                      : "Sign In to Enroll"}
                </button>

                <button className="w-full py-3 rounded-xl font-semibold text-sm text-gray-700 bg-gray-100 hover:bg-gray-200 transition-colors border border-gray-200">
                  Add to Wishlist
                </button>

                {/* Divider */}
                <hr className="my-5 border-gray-100" />

                {/* Stats */}
                <div className="space-y-3">
                  {[
                    { label: "Modules", value: course.total_modules },
                    {
                      label: "Students",
                      value: course.total_students?.toLocaleString(),
                    },
                    {
                      label: "Total Duration",
                      value: formatDuration(totalDuration),
                    },
                    { label: "Lessons", value: totalItems },
                    {
                      label: "Level",
                      value: capitalize(course.difficulty_level),
                    },
                    {
                      label: "Rating",
                      value: (
                        <span className="flex items-center gap-1">
                          <Star className="w-3.5 h-3.5 fill-yellow-400 text-yellow-400" />
                          4.5
                        </span>
                      ),
                    },
                  ].map(({ label, value }) => (
                    <div
                      key={label}
                      className="flex justify-between items-center text-sm"
                    >
                      <span className="text-gray-500">{label}</span>
                      <span className="font-semibold text-gray-900">
                        {value}
                      </span>
                    </div>
                  ))}
                </div>

                {/* Guarantee */}
                <div className="mt-5 p-3 bg-gray-50 rounded-lg border border-gray-100 text-center">
                  <p className="text-xs text-gray-500 flex items-center justify-center gap-1">
                    <Trophy className="w-3.5 h-3.5 text-yellow-500" />
                    30-day money-back guarantee
                  </p>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
