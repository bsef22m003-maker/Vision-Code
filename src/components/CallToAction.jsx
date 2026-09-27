import { NavLink } from "react-router-dom";

function CallToAction() {
  return (
    <div className="py-20 bg-gradient-to-r from-blue-600 to-purple-600 text-white scroll-fade-up">
      <div className="max-w-4xl mx-auto px-4 text-center">
        <h2 className="text-4xl font-bold mb-6">
          Ready to Start Your Coding Journey?
        </h2>
        <p className="text-xl mb-8 text-blue-100">
          Join thousands of developers learning to code with Vision-Code today.
        </p>
        <div className="flex justify-center gap-4 flex-wrap">
          <a
            href="signup.html"
            className="px-8 py-4 bg-white text-blue-600 rounded-lg font-semibold hover: shadow-xl transition"
          >
            Get Started Free
          </a>
          <NavLink
            to="/login"
            className="px-8 py-4 border-2 border-white text-white rounded-lg font-semibold hover:bg-white hover:text-blue-600 transition"
          >
            Browse Courses
          </NavLink>
        </div>
      </div>
    </div>
  );
}

export default CallToAction;
