import NavBar from "../components/NavBar";
import Footer from "../components/Footer";
import CommunityHeader from "../components/CommunityHeader";
import CommunitySidebar from "../components/CommunitySidebar";
import CommunityFeed from "../components/CommunityFeed";
import { NavLink } from "react-router-dom";
import { APP_NAME } from "../App";

function CommunityPage() {
  return (
    <div className="min-h-screen flex flex-col bg-gray-50">
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
              className="px-6 py-2 bg-gradient-to-r from-blue-600 to-purple-600 text-white rounded-lg hover: shadow-lg transition"
            >
              Sign Up
            </NavLink>
          </>
        }
      >
        <NavLink
          to="/courses"
          className="text-gray-700 hover:text-blue-600 font-medium transition"
        >
          Courses
        </NavLink>
        <NavLink
          to="/community"
          className={({ isActive }) =>
            `font-medium transition pb-1 ${
              isActive
                ? "text-blue-600 border-b-2 border-blue-600"
                : "text-gray-700 hover:text-blue-600"
            }`
          }
        >
          Community
        </NavLink>
        <NavLink
          to="/practice"
          className="text-gray-700 hover: text-blue-600 font-medium transition"
        >
          Practice
        </NavLink>
      </NavBar>

      <CommunityHeader />

      <div className="max-w-7xl mx-auto px-4 py-8 grid grid-cols-1 lg:grid-cols-4 gap-8">
        <CommunitySidebar />
        <CommunityFeed />
      </div>

      <Footer />
    </div>
  );
}

export default CommunityPage;
