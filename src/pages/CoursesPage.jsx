import Navbar from "../components/NavBar";
import Footer from "../components/Footer";
import CourseHero from "../components/CourseHero";
import CourseFilters from "../components/CourseFilters";
import CourseGrid from "../components/CourseGrid";
import NewsLetter from "../components/NewsLetter";
import { NavLink } from "react-router-dom";
import { APP_NAME } from "../App";

function CoursesPage() {
  return (
    <div className="min-h-screen flex flex-col bg-gray-50">
      <Navbar
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
      </Navbar>

      <CourseHero />
      {/* <CourseFilters /> */}
      <CourseGrid />

      <Footer />
    </div>
  );
}

export default CoursesPage;
