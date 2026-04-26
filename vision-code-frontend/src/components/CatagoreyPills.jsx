import { useState } from "react";

function CategoryPills() {
  const [activeCategory, setActiveCategory] = useState("All Courses");

  const categories = [
    "All Courses",
    "Frontend",
    "Backend",
    "DevOps",
    "Data Science",
    "Mobile",
  ];

  return (
    <div className="flex flex-wrap justify-center gap-3 mt-8">
      {categories.map((category) => (
        <button
          key={category}
          onClick={() => setActiveCategory(category)}
          className={`px-5 py-2 rounded-full font-semibold transition transform hover:scale-105 ${
            activeCategory === category
              ? "bg-white text-blue-600 shadow-lg"
              : "bg-white/10 backdrop-blur-sm hover:bg-white/20 border border-white/30"
          }`}
        >
          {category}
        </button>
      ))}
    </div>
  );
}

export default CategoryPills;
