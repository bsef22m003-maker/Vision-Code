import { NavLink } from "react-router-dom";

function IntroPanel({ newCourse = "React 18 Course", devNums = "50,000+" }) {
  return (
    <div className="scroll-slide-left">
      <span className="bg-blue-500/30 text-blue-50 py-1 px-3 rounded-full text-sm font-medium mb-4 inline-block border border-blue-400/30">
        New: {newCourse}
      </span>
      <h1 className="text-5xl font-bold mb-6 leading-tight">
        Master Coding with Interactive Lessons
      </h1>
      <p className="text-xl mb-8 text-blue-100">
        Join {devNums} developers learning via structured modules, hands-on
        practice, and a supportive community.
      </p>
      <div className="flex space-x-4">
        <NavLink
          to="/login"
          className="px-8 py-4 bg-white text-blue-600 rounded-lg font-semibold hover:shadow-xl transition"
        >
          Start Learning
        </NavLink>
        <NavLink
          to="/login"
          className="px-8 py-4 border-2 border-white text-white rounded-lg font-semibold hover:bg-white hover:text-blue-600 transition"
        >
          Join Community
        </NavLink>
      </div>
    </div>
  );
}

export default IntroPanel;
