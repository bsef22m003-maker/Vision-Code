import { useState, useEffect } from "react";
import { Link, useNavigate } from "react-router-dom";
import {
  LogOut,
  Settings,
  Plus,
  BookOpen,
  Users,
  TrendingUp,
  Eye,
  Edit,
  Trash2,
  AlertCircle,
  X,
} from "lucide-react";
import { API_URL } from "../constants";
import { useAuth } from "../context/FakeAuth";
import InstructorDashboardSkeleton from "../components/InstructorDashboardSkeleton";

// ─── Course Card ──────────────────────────────────────────────────────────────

function CourseCard({ course, onEdit, onDelete }) {
  const difficultyColors = {
    beginner: "bg-green-100 text-green-700",
    intermediate: "bg-purple-100 text-purple-700",
    advanced: "bg-orange-100 text-orange-700",
  };

  const difficultyIcons = {
    beginner: "📚",
    intermediate: "🚀",
    advanced: "⚡",
  };

  const level = course.difficulty_level?.toLowerCase();
  const bgColor = difficultyColors[level] || "bg-blue-100 text-blue-700";
  const icon = difficultyIcons[level] || "📖";

  return (
    <div className="bg-white rounded-xl border border-gray-200 p-5 hover:shadow-md transition-all group">
      <div className="flex items-start gap-4 mb-4">
        <div className="w-12 h-12 bg-gradient-to-br from-blue-500 to-purple-600 rounded-lg flex items-center justify-center text-xl flex-shrink-0 shadow-sm">
          {icon}
        </div>
        <div className="flex-1 min-w-0">
          <div className="flex items-start justify-between gap-2 mb-1">
            <h3 className="font-bold text-gray-900 text-sm group-hover:text-blue-600 transition line-clamp-1">
              {course.title}
            </h3>
            <span
              className={`text-xs px-2 py-0.5 rounded-full font-medium flex-shrink-0 ${bgColor}`}
            >
              {course.difficulty_level?.charAt(0).toUpperCase() +
                course.difficulty_level?.slice(1)}
            </span>
          </div>
          <p className="text-xs text-gray-500 line-clamp-2 mb-3">
            {course.description}
          </p>
        </div>
      </div>

      {/* Stats */}
      <div className="flex items-center justify-between text-xs text-gray-600 mb-4 pb-4 border-b border-gray-100">
        <div className="flex items-center gap-4">
          {course.duration_hours && (
            <span className="flex items-center gap-1">
              <BookOpen className="w-3.5 h-3.5" />
              {course.duration_hours}h
            </span>
          )}
          {course.enrolled_count && (
            <span className="flex items-center gap-1">
              <Users className="w-3.5 h-3.5" />
              {course.enrolled_count} students
            </span>
          )}
        </div>
      </div>

      {/* Actions */}
      <div className="flex gap-2">
        <Link
          to={`/instructor/course/${course.id}`}
          className="flex-1 px-3 py-2 bg-blue-50 text-blue-600 rounded-lg text-xs font-semibold hover:bg-blue-100 transition flex items-center justify-center gap-1.5"
        >
          <Eye className="w-3.5 h-3.5" /> View
        </Link>
        {/* <button
          onClick={() => onEdit(course)}
          className="px-3 py-2 border border-gray-200 text-gray-700 rounded-lg text-xs font-semibold hover:bg-gray-50 transition"
        >
          <Edit className="w-3.5 h-3.5" />
        </button> */}
        <button
          onClick={() => onDelete(course.id)}
          className="px-3 py-2 border border-red-200 text-red-600 rounded-lg text-xs font-semibold hover:bg-red-50 transition"
        >
          <Trash2 className="w-3.5 h-3.5" />
        </button>
      </div>
    </div>
  );
}

// ─── Main Instructor Dashboard ─────────────────────────────────────────────────

export default function InstructorDashboard() {
  const navigate = useNavigate();
  const { logout } = useAuth();
  const token = localStorage.getItem("access_token");

  // State
  const [instructorProfile, setInstructorProfile] = useState(null);
  const [courses, setCourses] = useState([]);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState("");
  const [successMessage, setSuccessMessage] = useState("");
  const [totalStudents, setTotalStudents] = useState(0);

  // Fetch instructor data
  useEffect(() => {
    if (!token) {
      navigate("/login");
      return;
    }
    fetchInstructorData();
  }, [token, navigate]);

  const fetchInstructorData = async () => {
    try {
      setIsLoading(true);
      setError("");

      const token = localStorage.getItem("access_token");
      const decodedToken = token ? JSON.parse(atob(token.split(".")[1])) : null;
      const userId = decodedToken?.user_id;

      if (!token) throw new Error("No authentication token found");
      if (!userId) throw new Error("User ID not found in token");
      const [profileRes, coursesRes] = await Promise.all([
        fetch(`${API_URL}/api/accounts/users/${userId}/`, {
          headers: {
            "Content-Type": "application/json",
            Authorization: `Bearer ${token}`,
          },
        }),

        fetch(`${API_URL}/api/courses/courses/?instructor=${userId}`, {
          headers: {
            Authorization: `Bearer ${token}`,
            "Content-Type": "application/json",
          },
        }),
      ]);

      if (!profileRes.ok) {
        console.error("Profile response:", profileRes.status);
        throw new Error(`Failed to fetch profile: ${profileRes.status}`);
      }
      if (!coursesRes.ok) {
        console.error("Courses response:", coursesRes.status);
        throw new Error(`Failed to fetch courses: ${coursesRes.status}`);
      }

      const [profile, coursesData] = await Promise.all([
        profileRes.json(),
        coursesRes.json(),
      ]);
      setInstructorProfile(profile);
      setCourses(
        Array.isArray(coursesData) ? coursesData : (coursesData?.results ?? []),
      );
      setTotalStudents(
        courses.reduce((sum, c) => sum + (c.total_students ?? 0), 0),
      );
    } catch (err) {
      console.error("Fetch error:", err);
      setError(err instanceof Error ? err.message : "Failed to load dashboard");
    } finally {
      setIsLoading(false);
    }
  };

  const handleDeleteCourse = async (courseId) => {
    if (!window.confirm("Are you sure you want to delete this course?")) {
      return;
    }

    try {
      const token = localStorage.getItem("access_token");
      const response = await fetch(`${API_URL}/api/courses/${courseId}/`, {
        method: "DELETE",
        headers: {
          Authorization: `Bearer ${token}`,
        },
      });

      if (!response.ok) throw new Error("Failed to delete course");

      setCourses((prev) => prev.filter((c) => c.id !== courseId));
      setSuccessMessage("Course deleted successfully!");
      setTimeout(() => setSuccessMessage(""), 3000);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to delete course");
    }
  };

  const handleLogout = () => {
    logout();
    navigate("/login");
  };

  const fullName = instructorProfile
    ? `${instructorProfile.first_name} ${instructorProfile.last_name}`.trim()
    : "Instructor";

  const initials = instructorProfile
    ? `${instructorProfile.first_name?.[0] ?? ""}${
        instructorProfile.last_name?.[0] ?? ""
      }`.toUpperCase()
    : "I";

  const totalHours = courses.reduce(
    (sum, c) => sum + (c.duration_hours ?? 0),
    0,
  );

  // ── Loading ────────────────────────────────────────────────────────────────

  if (isLoading) {
    return <InstructorDashboardSkeleton />;
  }

  return (
    <div className="min-h-screen bg-gray-50">
      {/* ── Navbar ─�� */}
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
              <div className="flex items-center gap-2">
                <div className="w-9 h-9 bg-gradient-to-r from-blue-600 to-purple-600 rounded-full flex items-center justify-center text-white font-bold text-sm">
                  {initials}
                </div>
                <div className="hidden md:block">
                  <p className="text-sm font-semibold text-gray-900 leading-none">
                    {fullName}
                  </p>
                  <p className="text-xs text-gray-500">Instructor</p>
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
        {/* ── Success Message ── */}
        {successMessage && (
          <div className="mb-6 p-4 bg-green-50 border border-green-200 rounded-xl text-green-700 flex items-center justify-between">
            <span className="font-medium">{successMessage}</span>
            <button
              onClick={() => setSuccessMessage("")}
              className="text-green-600 hover:text-green-800"
            >
              <X className="w-4 h-4" />
            </button>
          </div>
        )}

        {/* ── Error ── */}
        {error && (
          <div className="mb-6 p-4 bg-red-50 border border-red-200 rounded-xl text-red-700">
            <div className="flex items-start gap-3">
              <AlertCircle className="w-5 h-5 flex-shrink-0 mt-0.5" />
              <div>
                <p className="font-semibold mb-2">Error loading dashboard</p>
                <p className="text-sm mb-3">{error}</p>
                <button
                  onClick={fetchInstructorData}
                  className="px-4 py-2 bg-red-600 text-white rounded-lg text-sm font-medium hover:bg-red-700 transition"
                >
                  Retry
                </button>
              </div>
            </div>
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
                {courses.length > 0
                  ? `You have ${courses.length} course${
                      courses.length !== 1 ? "s" : ""
                    } published. Keep teaching!`
                  : "Start creating your first course today."}
              </p>
            </div>
            <Link
              to="/instructor/create-course"
              className="px-4 py-2 bg-white text-blue-600 rounded-xl font-semibold text-sm hover:bg-blue-50 transition flex items-center gap-2"
            >
              <Plus className="w-4 h-4" />
              New Course
            </Link>
          </div>
        </div>

        {/* ── Stats ── */}
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-8">
          {[
            {
              label: "Published",
              value: courses.length,
              icon: <BookOpen className="w-5 h-5 text-blue-600" />,
              bg: "bg-blue-50",
              sub: "total courses",
            },
            {
              label: "Total Students",
              value: totalStudents,
              icon: <Users className="w-5 h-5 text-purple-600" />,
              bg: "bg-purple-50",
              sub: "enrolled",
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

        {/* ── Main Content ── */}
        <div className="bg-white rounded-xl shadow-sm border border-gray-100">
          <div className="flex items-center justify-between p-6 border-b border-gray-100">
            <h2 className="text-xl font-bold text-gray-900">Your Courses</h2>
            <Link
              to="/instructor/create-course"
              className="px-4 py-2 bg-blue-600 text-white rounded-lg text-sm font-semibold hover:bg-blue-700 transition flex items-center gap-2"
            >
              <Plus className="w-4 h-4" />
              Add Course
            </Link>
          </div>

          <div className="p-6">
            {courses.length === 0 ? (
              <div className="text-center py-12">
                <BookOpen className="w-12 h-12 text-gray-200 mx-auto mb-3" />
                <p className="text-gray-500 mb-4">
                  No courses yet. Start by creating your first course!
                </p>
                <Link
                  to="/instructor/create-course"
                  className="px-5 py-2 bg-blue-600 text-white rounded-lg text-sm font-semibold hover:bg-blue-700 transition inline-flex items-center gap-2"
                >
                  <Plus className="w-4 h-4" />
                  Create First Course
                </Link>
              </div>
            ) : (
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-5">
                {courses.map((course) => (
                  <CourseCard
                    key={course.id}
                    course={course}
                    onEdit={(course) => {
                      navigate(`/instructor/edit-course/${course.id}`);
                    }}
                    onDelete={handleDeleteCourse}
                  />
                ))}
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
