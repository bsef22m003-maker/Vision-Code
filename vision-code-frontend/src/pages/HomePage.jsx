import { NavLink } from "react-router-dom";
import CallToAction from "../components/CallToAction";
import Features from "../components/Features";
import Footer from "../components/Footer";
import HeroSection from "../components/HeroSection";
import LearningJourney from "../components/LearningJourney";
import NavBar from "../components/NavBar";
import SkillsSection from "../components/SkillsSection";
import Stats from "../components/Stats";
import useScrollAnimation from "../hooks/useScrollAnimation";
export const APP_NAME = "Vision-Code";

function HomePage() {
  useScrollAnimation();
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
          to="/community"
          className="text-gray-700 hover:text-blue-600 font-medium transition"
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
      <HeroSection />
      <Stats />
      <Features />
      <SkillsSection />
      <LearningJourney />
      <CallToAction />
      <Footer />
    </div>
  );
}

export default HomePage;
