import { useEffect, useState } from "react";
import NavBar from "../components/NavBar";
import Footer from "../components/Footer";
import StatsCard from "../components/Instructor/StatsCard";
import CourseCard from "../components/Instructor/InsCourseCard";
import CreateCourseModal from "../components/Instructor/CreateCourseModal";
import { NavLink, useNavigate } from "react-router-dom";
import { getTokenData } from "../utils/tokenUtils";
import { useAuth } from "../context/FakeAuth";
import Button from "../components/Button";
const stats = [
  {
    icon: (
      <svg
        className="w-6 h-6"
        fill="none"
        stroke="currentColor"
        viewBox="0 0 24 24"
      >
        <path
          strokeLinecap="round"
          strokeLinejoin="round"
          strokeWidth="2"
          d="M12 4.354a4 4 0 110 5.292M15 21H3v-1a6 6 0 0112 0v1zm0 0h6v-1a6 6 0 00-9-5.197M13 7a4 4 0 11-8 0 4 4 0 018 0z"
        />
      </svg>
    ),
    gradient: "from-blue-500 to-blue-600",
    value: "6,095",
    label: "Total Students",
    change: "+12. 5%",
    positive: true,
  },
  {
    icon: (
      <svg
        className="w-6 h-6"
        fill="none"
        stroke="currentColor"
        viewBox="0 0 24 24"
      >
        <path
          strokeLinecap="round"
          strokeLinejoin="round"
          strokeWidth="2"
          d="M12 8c-1.657 0-3 . 895-3 2s1.343 2 3 2 3 .895 3 2-1.343 2-3 2m0-8c1.11 0 2.08.402 2.599 1M12 8V7m0 1v8m0 0v1m0-1c-1.11 0-2.08-.402-2.599-1M21 12a9 9 0 11-18 0 9 9 0 0118 0z"
        />
      </svg>
    ),
    gradient: "from-green-500 to-green-600",
    value: "$22,650",
    label: "Total Revenue",
    change: "+8.2%",
    positive: true,
  },
  {
    icon: (
      <svg
        className="w-6 h-6"
        fill="none"
        stroke="currentColor"
        viewBox="0 0 24 24"
      >
        <path
          strokeLinecap="round"
          strokeLinejoin="round"
          strokeWidth="2"
          d="M12 6. 253v13m0-13C10.832 5.477 9.246 5 7.5 5S4.168 5.477 3 6.253v13C4.168 18.477 5.754 18 7.5 18s3.332.477 4.5 1.253m0-13C13.168 5.477 14.754 5 16.5 5c1.747 0 3.332.477 4.5 1.253v13C19.832 18.477 18.247 18 16.5 18c-1.746 0-3.332.477-4.5 1.253"
        />
      </svg>
    ),
    gradient: "from-purple-500 to-purple-600",
    value: "2",
    label: "Active Courses",
    change: "+1",
    positive: true,
  },
  {
    icon: (
      <svg className="w-6 h-6" fill="currentColor" viewBox="0 0 20 20">
        <path d="M9.049 2.927c.3-. 921 1.603-.921 1.902 0l1.07 3.292a1 1 0 00.95.69h3.462c.969 0 1.371 1.24.588 1.81l-2.8 2.034a1 1 0 00-.364 1.118l1.07 3.292c.3.921-.755 1.688-1.54 1.118l-2.8-2.034a1 1 0 00-1.175 0l-2.8 2.034c-.784.57-1.838-.197-1.539-1.118l1.07-3.292a1 1 0 00-.364-1.118L2.98 8.72c-.783-.57-. 38-1.81.588-1.81h3.461a1 1 0 00.951-.69l1.07-3.292z" />
      </svg>
    ),
    gradient: "from-yellow-500 to-orange-500",
    value: "4. 85",
    label: "Avg. Rating",
    change: "+0.15",
    positive: true,
  },
];

const demoCourses = [
  {
    id: 1,
    title: "Modern JavaScript Bootcamp",
    icon: "JS",
    gradient: "from-yellow-400 to-orange-500",
    status: "published",
    students: 3245,
    revenue: 12450,
    rating: 4.8,
    reviews: 1200,
    modules: 12,
    lastUpdated: "2 days ago",
  },
  {
    id: 2,
    title: "React 18 Mastery",
    icon: "React",
    gradient: "from-cyan-500 to-blue-600",
    status: "published",
    students: 2850,
    revenue: 10200,
    rating: 4.9,
    reviews: 850,
    modules: 20,
    lastUpdated: "1 week ago",
  },
  {
    id: 3,
    title: "Advanced TypeScript Patterns",
    icon: "TS",
    gradient: "from-blue-500 to-indigo-600",
    status: "draft",
    students: 0,
    revenue: 0,
    rating: null,
    reviews: 0,
    modules: 5,
    lastUpdated: "3 hours ago",
  },
];

function InstructorDashboard() {
  const [showCreateModal, setShowCreateModal] = useState(false);
  const [filterTab, setFilterTab] = useState("all");
  const [instructor, setInstructor] = useState(null);
  const [courses, setCourses] = useState(demoCourses);
  const { userId } = getTokenData(localStorage.getItem("access_token"));
  const { logout } = useAuth();
  const navigate = useNavigate();
  const handleLogout = () => {
    logout();
    navigate("/");
  };
  useEffect(() => {
    const fetchCourses = async () => {
      try {
        const response = await fetch();
        if (!response.ok) {
          throw new Error("Failed to fetch courses");
        }
        const data = await response.json();
        setCourses(data);
      } catch (error) {
        console.error("Error fetching courses:", error);
      }
    };
    const fetchUserData = async () => {
      try {
        const response = await fetch(
          `http://localhost:8000/api/users/${userId}`
        );
        if (!response.ok) {
          throw new Error("Failed to fetch user data");
        }
        const data = await response.json();
        setInstructor(data);
      } catch (error) {
        console.error("Error fetching user data:", error);
      }
    };

    fetchUserData();
  }, [userId]);
  console.log("Instructor Data:", instructor);

  const filteredCourses = courses.filter((course) => {
    if (filterTab === "all") return true;
    return course.status === filterTab;
  });

  const allCount = courses.length;
  const publishedCount = courses.filter((c) => c.status === "published").length;
  const draftCount = courses.filter((c) => c.status === "draft").length;

  return (
    <div className="bg-gray-50 text-gray-800 font-sans">
      {/* Navbar */}
      <NavBar
        authButtons={
          <div className="hidden md:flex items-center space-x-4">
            <button className="p-2 text-gray-600 hover:text-blue-600 transition">
              <svg
                className="w-6 h-6"
                fill="none"
                stroke="currentColor"
                viewBox="0 0 24 24"
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth="2"
                  d="M15 17h5l-1.405-1.405A2.032 2.032 0 0118 14.158V11a6.002 6.002 0 00-4-5.659V5a2 2 0 10-4 0v.341C7.67 6.165 6 8.388 6 11v3.159c0 .538-.214 1.055-.595 1.436L4 17h5m6 0v1a3 3 0 11-6 0v-1m6 0H9"
                />
              </svg>
            </button>
            <div className="flex items-center space-x-3">
              <div className="w-10 h-10 bg-gradient-to-r from-blue-600 to-purple-600 rounded-full flex items-center justify-center text-white font-bold">
                {instructor
                  ? `${instructor.first_name
                      .charAt(0)
                      .toUpperCase()} ${instructor.last_name
                      .charAt(0)
                      .toUpperCase()}`
                  : "L"}
              </div>
              <div className="hidden lg:block">
                <div className="text-sm font-semibold text-gray-900">
                  {instructor ? `${instructor.first_name}` : "Loading..."}
                </div>
                <div className="text-xs text-gray-500">Instructor</div>
              </div>
            </div>
            <Button variant="ghost" size="sm" onClick={handleLogout}>
              Logout
            </Button>
          </div>
        }
      >
        <NavLink
          to="/instructor"
          className="text-blue-600 font-medium border-b-2 border-blue-600 pb-1"
        >
          My Courses
        </NavLink>
        <NavLink
          to="/instructor/analytics"
          className="text-gray-700 hover:text-blue-600 font-medium transition"
        >
          Analytics
        </NavLink>
        <NavLink
          to="/instructor/earnings"
          className="text-gray-700 hover:text-blue-600 font-medium transition"
        >
          Earnings
        </NavLink>
        <NavLink
          to="/instructor/students"
          className="text-gray-700 hover:text-blue-600 font-medium transition"
        >
          Students
        </NavLink>
      </NavBar>

      {/* Hero Section */}
      <div className="relative bg-gradient-to-br from-blue-600 via-purple-600 to-indigo-700 text-white py-16 mb-8 overflow-hidden">
        <div className="absolute top-0 right-0 w-96 h-96 bg-white/10 rounded-full blur-3xl animate-pulse"></div>
        <div
          className="absolute bottom-0 left-0 w-96 h-96 bg-purple-500/20 rounded-full blur-3xl animate-pulse"
          style={{ animationDelay: "1s" }}
        ></div>

        <div className="max-w-7xl mx-auto px-4 relative z-10">
          <div className="flex justify-between items-center">
            <div>
              <h1 className="text-4xl font-bold mb-2">
                Welcome back, John! 👋
              </h1>
              <p className="text-blue-100 text-lg">
                Manage your courses and track your impact
              </p>
            </div>
            <button
              onClick={() => setShowCreateModal(true)}
              className="bg-white text-blue-600 hover:shadow-xl hover: scale-105 px-6 py-3 rounded-lg font-semibold transition-all duration-300 flex items-center gap-2"
            >
              <svg
                className="w-5 h-5"
                fill="none"
                stroke="currentColor"
                viewBox="0 0 24 24"
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth="2"
                  d="M12 4v16m8-8H4"
                />
              </svg>
              Create New Course
            </button>
          </div>
        </div>
      </div>

      {/* Stats Section */}
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 -mt-8 mb-12 relative z-10">
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
          {stats.map((stat, index) => (
            <StatsCard key={index} {...stat} />
          ))}
        </div>
      </div>

      {/* Courses Section */}
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 pb-20">
        <div className="flex justify-between items-center mb-8">
          <div>
            <h2 className="text-2xl font-bold text-gray-900">Your Courses</h2>
            <p className="text-gray-600 mt-1">
              Manage and track your course performance
            </p>
          </div>

          {/* Filter Tabs */}
          <div className="flex gap-2">
            <button
              onClick={() => setFilterTab("all")}
              className={`px-4 py-2 rounded-lg font-medium text-sm ${
                filterTab === "all"
                  ? "bg-blue-600 text-white"
                  : "bg-white text-gray-700 border border-gray-300 hover:bg-gray-50"
              }`}
            >
              All ({allCount})
            </button>
            <button
              onClick={() => setFilterTab("published")}
              className={`px-4 py-2 rounded-lg font-medium text-sm ${
                filterTab === "published"
                  ? "bg-blue-600 text-white"
                  : "bg-white text-gray-700 border border-gray-300 hover:bg-gray-50"
              }`}
            >
              Published ({publishedCount})
            </button>
            <button
              onClick={() => setFilterTab("draft")}
              className={`px-4 py-2 rounded-lg font-medium text-sm ${
                filterTab === "draft"
                  ? "bg-blue-600 text-white"
                  : "bg-white text-gray-700 border border-gray-300 hover:bg-gray-50"
              }`}
            >
              Draft ({draftCount})
            </button>
          </div>
        </div>

        {/* Courses Grid */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
          {filteredCourses.map((course) => (
            <CourseCard key={course.id} course={course} />
          ))}

          {/* Add New Course Card */}
          <div
            onClick={() => setShowCreateModal(true)}
            className="bg-white rounded-xl shadow-md border-2 border-dashed border-gray-300 hover:border-blue-500 transition-all duration-300 cursor-pointer group flex flex-col items-center justify-center p-8 min-h-[400px]"
          >
            <div className="w-20 h-20 bg-blue-100 rounded-full flex items-center justify-center mb-4 group-hover:bg-blue-600 transition-all duration-300">
              <svg
                className="w-10 h-10 text-blue-600 group-hover:text-white transition-all duration-300"
                fill="none"
                stroke="currentColor"
                viewBox="0 0 24 24"
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth="2"
                  d="M12 4v16m8-8H4"
                />
              </svg>
            </div>
            <h3 className="text-xl font-bold text-gray-900 mb-2 group-hover:text-blue-600 transition">
              Create New Course
            </h3>
            <p className="text-gray-600 text-center text-sm">
              Start building your next amazing course
            </p>
          </div>
        </div>
      </div>

      {/* Create Course Modal */}
      {showCreateModal && (
        <CreateCourseModal onClose={() => setShowCreateModal(false)} />
      )}

      {/* Footer */}
      <Footer />
    </div>
  );
}

export default InstructorDashboard;
