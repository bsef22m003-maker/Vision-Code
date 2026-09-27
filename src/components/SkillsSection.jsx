import { NavLink } from "react-router-dom";
import SkillCard from "./SkillCard";

function SkillsSection() {
  const skillsData = [
    {
      gradientFrom: "from-orange-500",
      gradientTo: "to-pink-600",
      glowFrom: "from-orange-400",
      glowTo: "to-pink-500",
      title: "Frontend Development",
      skills: [
        "HTML5 & CSS3",
        "JavaScript & TypeScript",
        "React & Vue",
        "Responsive Design",
      ],
      delay: "delay-100",
      icon: (
        <svg
          fill="none"
          stroke="currentColor"
          strokeWidth="2"
          viewBox="0 0 24 24"
          className="w-full h-full"
        >
          <path
            strokeLinecap="round"
            strokeLinejoin="round"
            d="M9.75 17L9 20l-1 1h8l-1-1-.75-3M3 13h18M5 17h14a2 2 0 002-2V5a2 2 0 00-2-2H5a2 2 0 00-2 2v10a2 2 0 002 2z"
          />
        </svg>
      ),
    },
    {
      gradientFrom: "from-blue-500",
      gradientTo: "to-indigo-600",
      glowFrom: "from-blue-400",
      glowTo: "to-indigo-500",
      title: "Backend Development",
      skills: [
        "Node.js & Express",
        "Python & Django",
        "REST APIs",
        "Database Design",
      ],
      delay: "delay-200",
      icon: (
        <svg
          fill="none"
          stroke="currentColor"
          strokeWidth="2"
          viewBox="0 0 24 24"
          className="w-full h-full"
        >
          <path
            strokeLinecap="round"
            strokeLinejoin="round"
            d="M5 12h14M5 12a2 2 0 01-2-2V6a2 2 0 012-2h14a2 2 0 012 2v4a2 2 0 01-2 2M5 12a2 2 0 00-2 2v4a2 2 0 002 2h14a2 2 0 002-2v-4a2 2 0 00-2-2m-2-4h. 01M17 16h.01"
          />
        </svg>
      ),
    },
    {
      gradientFrom: "from-emerald-500",
      gradientTo: "to-cyan-600",
      glowFrom: "from-emerald-400",
      glowTo: "to-cyan-500",
      title: "Data Science",
      skills: [
        "Python for Data",
        "Machine Learning",
        "Data Visualization",
        "SQL & Analytics",
      ],
      delay: "delay-300",
      icon: (
        <svg fill="currentColor" viewBox="0 0 24 24" className="w-full h-full">
          <path d="M19 3H5c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h14c1.1 0 2-.9 2-2V5c0-1.1-.9-2-2-2zM9 17H7v-7h2v7zm4 0h-2V7h2v10zm4 0h-2v-4h2v4z" />
        </svg>
      ),
    },
    {
      gradientFrom: "from-purple-500",
      gradientTo: "to-pink-600",
      glowFrom: "from-purple-400",
      glowTo: "to-pink-500",
      title: "Data Structures & Algorithms",
      skills: [
        "Arrays & Linked Lists",
        "Trees & Graphs",
        "Sorting & Searching",
        "Dynamic Programming",
      ],
      delay: "delay-400",
      icon: (
        <svg
          fill="none"
          stroke="currentColor"
          strokeWidth="2"
          viewBox="0 0 24 24"
          className="w-full h-full"
        >
          <path
            strokeLinecap="round"
            strokeLinejoin="round"
            d="M20 7l-8-4-8 4m16 0l-8 4m8-4v10l-8 4m0-10L4 7m8 4v10M4 7v10l8 4"
          />
        </svg>
      ),
    },
  ];

  return (
    <div className="py-20 bg-white">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg: px-8">
        <div className="text-center mb-16 scroll-fade-up">
          <h2 className="text-4xl font-bold text-gray-900 mb-4">
            Master In-Demand Skills
          </h2>
          <p className="text-xl text-gray-600">
            Build expertise across modern development technologies
          </p>
        </div>

        <div className="grid md:grid-cols-2 lg:grid-cols-4 gap-6">
          {skillsData.map((skill, index) => (
            <SkillCard
              key={index}
              gradientFrom={skill.gradientFrom}
              gradientTo={skill.gradientTo}
              glowFrom={skill.glowFrom}
              glowTo={skill.glowTo}
              icon={skill.icon}
              title={skill.title}
              skills={skill.skills}
              delay={skill.delay}
            />
          ))}
        </div>

        <div className="text-center mt-12 scroll-fade-up">
          <NavLink
            to="/login"
            className="inline-block px-8 py-3 bg-gradient-to-r from-blue-600 to-purple-600 text-white rounded-lg font-semibold hover: shadow-lg transition"
          >
            Explore All Skills
          </NavLink>
        </div>
      </div>
    </div>
  );
}

export default SkillsSection;
