import { useState } from "react";
import SearchBar from "./SearchBar";
import CategoryPills from "./CatagoreyPills";

function CourseHero() {
  return (
    <div className="relative bg-gradient-to-br from-blue-600 via-purple-600 to-indigo-700 text-white py-20 mb-12 overflow-hidden">
      {/* Animated background elements */}
      <div className="absolute top-0 right-0 w-96 h-96 bg-white/10 rounded-full blur-3xl animate-pulse"></div>
      <div
        className="absolute bottom-0 left-0 w-96 h-96 bg-purple-500/20 rounded-full blur-3xl animate-pulse"
        style={{ animationDelay: "1s" }}
      ></div>

      <div className="max-w-7xl mx-auto px-4 text-center relative z-10">
        <div className="inline-block mb-4">
          <span className="px-4 py-2 bg-white/20 backdrop-blur-sm rounded-full text-sm font-semibold border border-white/30">
            🎓 500+ Courses Available
          </span>
        </div>
        <h1 className="text-5xl font-bold mb-4">Explore Our Course Catalog</h1>
        <p className="text-xl text-blue-100 mb-8 max-w-2xl mx-auto">
          Master in-demand skills with expert-led courses. Start learning today
          and advance your career.
        </p>

        <SearchBar />
        <CategoryPills />
      </div>
    </div>
  );
}

export default CourseHero;
