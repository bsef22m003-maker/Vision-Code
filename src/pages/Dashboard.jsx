import { useEffect, useState } from "react";
import { Link, NavLink, useNavigate } from "react-router-dom";
import {
  BookOpen,
  Star,
  Clock,
  LogOut,
  Settings,
  ArrowRight,
  Zap,
  Trophy,
  Users,
  TrendingUp,
  CheckCircle,
} from "lucide-react";
import { API_URL } from "../constants";
import { useAuth } from "../context/FakeAuth";
import DashboardSkeleton from "../components/DashboardSkeleton";

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

const getDifficultyColor = (level) => {
  switch (level) {
    case "beginner":
      return "bg-green-100 text-green-700";
    case "intermediate":
      return "bg-purple-100 text-purple-700";
    case "advanced":
      return "bg-orange-100 text-orange-700";
    default:
      return "bg-gray-100 text-gray-700";
  }
};

const getProgressColor = (progress) => {
  if (progress >= 75) return "from-green-500 to-emerald-500";
  if (progress >= 50) return "from-blue-500 to-indigo-500";
  if (progress >= 25) return "from-yellow-500 to-orange-500";
  return "from-red-500 to-pink-500";
};

// ─── Enrolled Course Card ─────────────────────────────────────────────────────

function EnrolledCourseCard({ course }) {
  const navigate = useNavigate();
  const progress = course.progress ?? 0;
  const level = course.difficulty_level?.toLowerCase();
  const gradient = levelGradients[level] || "from-blue-500 to-indigo-600";
  const icon = levelIcons[level] || "📖";

  return (
    <div
      onClick={() => navigate(`/student/course/${course.id}`)}
      className="border border-gray-200 rounded-xl p-4 hover:shadow-md transition-all cursor-pointer group hover:border-blue-300"
    >
      <div className="flex items-start gap-4">
        <div
          className={`w-14 h-14 bg-gradient-to-br ${gradient} rounded-xl flex items-center justify-center text-2xl flex-shrink-0 shadow-sm`}
        >
          {icon}
        </div>
        <div className="flex-1 min-w-0">
          <div className="flex items-start justify-between mb-1 gap-2">
            <h3 className="font-bold text-gray-900 text-sm group-hover:text-blue-600 transition line-clamp-1">
              {course.title}
            </h3>
            <span
              className={`text-xs px-2 py-0.5 rounded-full font-medium flex-shrink-0 ${getDifficultyColor(level)}`}
            >
              {course.difficulty_level?.charAt(0).toUpperCase() +
                course.difficulty_level?.slice(1)}
            </span>
          </div>
          <p className="text-xs text-gray-500 mb-3 line-clamp-1">
            {course.description}
          </p>
          <div className="w-full bg-gray-100 rounded-full h-1.5 mb-1">
            <div
              className={`bg-gradient-to-r ${getProgressColor(progress)} h-1.5 rounded-full transition-all`}
              style={{ width: `${progress}%` }}
            />
          </div>
          <div className="flex items-center justify-between text-xs text-gray-500">
            <span>{progress}% complete</span>
            {course.enrolled_at && (
              <span>
                Enrolled {new Date(course.enrolled_at).toLocaleDateString()}
              </span>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}

// ─── Main Dashboard ───────────────────────────────────────────────────────────

export default function Dashboard() {
  const navigate = useNavigate();
  const [enrolledCourses, setEnrolledCourses] = useState([]);
  const [studentProfile, setStudentProfile] = useState(null);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState("");
  const [activeTab, setActiveTab] = useState("enrolled");
  const { isAuthenticated, logout } = useAuth();
  const token = localStorage.getItem("access_token");
  useEffect(() => {
    if (!isAuthenticated || !token) {
      navigate("/login");
      return;
    }
    fetchStudentData();
  }, [token]);

  const fetchStudentData = async () => {
    try {
      setIsLoading(true);
      setError("");

      const token = localStorage.getItem("access_token");
      const decodedToken = token ? JSON.parse(atob(token.split(".")[1])) : null;
      const userId = decodedToken?.user_id;

      if (!token) throw new Error("No authentication token found");
      const [profileRes, coursesRes] = await Promise.all([
        fetch(`${API_URL}/api/accounts/users/${userId}`, {
          headers: {
            "Content-Type": "application/json",
            Authorization: `Bearer ${token}`,
          },
        }),
        fetch(`${API_URL}/api/enrollments/my-courses/`, {
          headers: {
            Authorization: `Bearer ${token}`,
            "Content-Type": "application/json",
          },
        }),
      ]);

      if (!profileRes.ok)
        throw new Error(`Failed to fetch profile: ${profileRes.status}`);
      if (!coursesRes.ok)
        throw new Error(`Failed to fetch courses: ${coursesRes.status}`);

      const [profile, courses] = await Promise.all([
        profileRes.json(),
        coursesRes.json(),
      ]);

      setStudentProfile(profile);
      setEnrolledCourses(
        Array.isArray(courses) ? courses : (courses?.results ?? []),
      );
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to load dashboard");
    } finally {
      setIsLoading(false);
    }
  };

  const handleLogout = () => {
    logout();
    navigate("/login");
  };

  const fullName = studentProfile
    ? `${studentProfile.first_name} ${studentProfile.last_name}`.trim()
    : "Student";

  const initials = studentProfile
    ? `${studentProfile.first_name?.[0] ?? ""}${studentProfile.last_name?.[0] ?? ""}`.toUpperCase()
    : "S";

  const completedCourses = enrolledCourses.filter((c) => c.completed);
  const inProgressCourses = enrolledCourses.filter((c) => !c.completed);
  const avgProgress =
    enrolledCourses.length > 0
      ? Math.round(
          enrolledCourses.reduce((sum, c) => sum + (c.progress ?? 0), 0) /
            enrolledCourses.length,
        )
      : 0;

  // ── Loading ────────────────────────────────────────────────────────────────
  if (isLoading) {
    return <DashboardSkeleton />;
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
            <NavLink
              to="https://visionocr.up.railway.app/"
              className={({ isActive }) =>
                `font-medium transition pb-1 ${
                  isActive
                    ? "text-blue-600 border-b-2 border-blue-600"
                    : "text-gray-700 hover:text-blue-600"
                }`
              }
            >
              Practice
            </NavLink>
            <div className="flex items-center gap-3">
              <div className="flex items-center gap-2">
                <div className="w-9 h-9 bg-gradient-to-r from-blue-600 to-purple-600 rounded-full flex items-center justify-center text-white font-bold text-sm">
                  {initials}
                </div>
                <div className="hidden md:block">
                  <p className="text-sm font-semibold text-gray-900 leading-none">
                    {fullName}
                  </p>
                  <p className="text-xs text-gray-500">
                    {studentProfile?.email}
                  </p>
                </div>
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
        {/* ── Error ── */}
        {error && (
          <div className="mb-6 p-4 bg-red-50 border border-red-200 rounded-xl text-red-700">
            <p className="font-semibold mb-1">Error loading dashboard</p>
            <p className="text-sm mb-3">{error}</p>
            <button
              onClick={fetchStudentData}
              className="px-4 py-2 bg-red-600 text-white rounded-lg text-sm font-medium hover:bg-red-700 transition"
            >
              Retry
            </button>
          </div>
        )}

        {/* ── Welcome Banner ── */}
        <div className="bg-gradient-to-r from-blue-600 via-purple-600 to-indigo-600 rounded-2xl p-8 mb-8 text-white relative overflow-hidden">
          <div className="absolute top-0 right-0 w-64 h-64 bg-white/5 rounded-full -translate-y-1/2 translate-x-1/2" />
          <div className="absolute bottom-0 left-1/2 w-48 h-48 bg-white/5 rounded-full translate-y-1/2" />
          <div className="relative z-10 flex items-center justify-between flex-wrap gap-4">
            <div>
              <h1 className="text-3xl font-bold mb-1">
                Welcome back, {fullName.split(" ")[0]}! 👋
              </h1>
              <p className="text-blue-100 text-sm">
                {enrolledCourses.length > 0
                  ? `You have ${inProgressCourses.length} course${inProgressCourses.length !== 1 ? "s" : ""} in progress. Keep it up!`
                  : "Start your learning journey today."}
              </p>
            </div>
            <div className="flex gap-3">
              <Link
                to="/courses"
                className="px-4 py-2 bg-white text-blue-600 rounded-xl font-semibold text-sm hover:bg-blue-50 transition"
              >
                Browse Courses
              </Link>
            </div>
          </div>
        </div>

        {/* ── Stats ── */}
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-8">
          {[
            {
              label: "Enrolled",
              value: enrolledCourses.length,
              icon: <BookOpen className="w-5 h-5 text-blue-600" />,
              bg: "bg-blue-50",
              sub: "total courses",
            },
            {
              label: "In Progress",
              value: inProgressCourses.length,
              icon: <Clock className="w-5 h-5 text-orange-600" />,
              bg: "bg-orange-50",
              sub: "active now",
            },
            {
              label: "Completed",
              value: completedCourses.length,
              icon: <CheckCircle className="w-5 h-5 text-green-600" />,
              bg: "bg-green-50",
              sub: "finished",
            },
          ].map(({ label, value, icon, bg, sub }) => (
            <div
              key={label}
              className="bg-white rounded-xl p-5 shadow-sm border border-gray-100 hover:shadow-md transition"
            >
              <div className="flex items-center justify-between mb-3">
                <span className="text-sm text-gray-500 font-medium">
                  {label}
                </span>
                <div
                  className={`w-9 h-9 ${bg} rounded-lg flex items-center justify-center`}
                >
                  {icon}
                </div>
              </div>
              <p className="text-3xl font-bold text-gray-900">{value}</p>
              <p className="text-xs text-gray-400 mt-1">{sub}</p>
            </div>
          ))}
        </div>

        {/* ── Main Grid ── */}
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
          {/* Left — Courses */}
          <div className="lg:col-span-2 space-y-6">
            {/* Tabs + Course List */}
            <div className="bg-white rounded-xl shadow-sm border border-gray-100">
              <div className="flex items-center justify-between p-6 pb-0">
                <h2 className="text-xl font-bold text-gray-900">My Courses</h2>
                <Link
                  to="/courses"
                  className="text-blue-600 hover:text-blue-700 text-sm font-medium flex items-center gap-1"
                >
                  Browse more <ArrowRight className="w-3.5 h-3.5" />
                </Link>
              </div>

              {/* Tabs */}
              <div className="flex gap-4 px-6 mt-4 border-b border-gray-100">
                {[
                  {
                    key: "enrolled",
                    label: `In Progress (${inProgressCourses.length})`,
                  },
                  {
                    key: "completed",
                    label: `Completed (${completedCourses.length})`,
                  },
                ].map((tab) => (
                  <button
                    key={tab.key}
                    onClick={() => setActiveTab(tab.key)}
                    className={`pb-3 text-sm font-semibold transition-colors border-b-2 ${
                      activeTab === tab.key
                        ? "text-blue-600 border-blue-600"
                        : "text-gray-500 border-transparent hover:text-gray-700"
                    }`}
                  >
                    {tab.label}
                  </button>
                ))}
              </div>

              <div className="p-6">
                {activeTab === "enrolled" && (
                  <>
                    {inProgressCourses.length === 0 ? (
                      <div className="text-center py-10">
                        <BookOpen className="w-12 h-12 text-gray-200 mx-auto mb-3" />
                        <p className="text-gray-500 mb-4">
                          No courses in progress
                        </p>
                        <Link
                          to="/courses"
                          className="px-5 py-2 bg-blue-600 text-white rounded-lg text-sm font-semibold hover:bg-blue-700 transition"
                        >
                          Browse Courses
                        </Link>
                      </div>
                    ) : (
                      <div className="space-y-3">
                        {inProgressCourses.map((course) => (
                          <EnrolledCourseCard key={course.id} course={course} />
                        ))}
                      </div>
                    )}
                  </>
                )}

                {activeTab === "completed" && (
                  <>
                    {completedCourses.length === 0 ? (
                      <div className="text-center py-10">
                        <Star className="w-12 h-12 text-gray-200 mx-auto mb-3" />
                        <p className="text-gray-500">
                          No completed courses yet
                        </p>
                        <p className="text-xs text-gray-400 mt-1">
                          Keep learning to see your completions here!
                        </p>
                      </div>
                    ) : (
                      <div className="space-y-3">
                        {completedCourses.map((course) => (
                          <EnrolledCourseCard key={course.id} course={course} />
                        ))}
                      </div>
                    )}
                  </>
                )}
              </div>
            </div>
          </div>

          {/* Right — Sidebar */}
          <div className="space-y-6">
            {/* Profile card */}
            <div className="bg-white rounded-xl shadow-sm border border-gray-100 p-6">
              <div className="flex items-center gap-3 mb-4">
                <div className="w-14 h-14 bg-gradient-to-br from-blue-500 to-purple-600 rounded-xl flex items-center justify-center text-white font-bold text-xl">
                  {initials}
                </div>
                <div>
                  <p className="font-bold text-gray-900">{fullName}</p>
                  <p className="text-xs text-gray-500">
                    {studentProfile?.email}
                  </p>
                  <span className="text-xs bg-blue-100 text-blue-700 px-2 py-0.5 rounded-full font-medium">
                    {studentProfile?.role ?? "Student"}
                  </span>
                </div>
              </div>
              <div className="grid grid-cols-2 gap-3 text-center">
                <div className="bg-gray-50 rounded-lg p-3">
                  <p className="text-2xl font-bold text-gray-900">
                    {enrolledCourses.length}
                  </p>
                  <p className="text-xs text-gray-500">Courses</p>
                </div>
                <div className="bg-gray-50 rounded-lg p-3">
                  <p className="text-2xl font-bold text-gray-900">
                    {completedCourses.length}
                  </p>
                  <p className="text-xs text-gray-500">Completed</p>
                </div>
              </div>
            </div>

            {/* Daily Challenge */}

            {/* Achievements */}
            <div className="bg-white rounded-xl shadow-sm border border-gray-100 p-6">
              <h3 className="font-bold text-gray-900 mb-4 flex items-center gap-2">
                <Trophy className="w-4 h-4 text-yellow-500" />
                Achievements
              </h3>
              <div className="space-y-3">
                {[
                  {
                    emoji: "🏆",
                    title: "Fast Learner",
                    desc: "Completed 5 lessons in one day",
                  },
                  {
                    emoji: "🎓",
                    title: "First Enrollment",
                    desc: `Enrolled in ${enrolledCourses[0]?.title ?? "a course"}`,
                  },
                  {
                    emoji: "⚡",
                    title: "Getting Started",
                    desc: "Joined Vision-Code",
                  },
                ].map(({ emoji, title, desc }) => (
                  <div key={title} className="flex items-center gap-3">
                    <div className="w-10 h-10 bg-yellow-50 rounded-lg flex items-center justify-center text-xl flex-shrink-0">
                      {emoji}
                    </div>
                    <div>
                      <p className="text-sm font-semibold text-gray-900">
                        {title}
                      </p>
                      <p className="text-xs text-gray-500">{desc}</p>
                    </div>
                  </div>
                ))}
              </div>
            </div>

            {/* Community */}
          </div>
        </div>
      </div>
    </div>
  );
}
