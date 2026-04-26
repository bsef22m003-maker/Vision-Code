import { NavLink } from "react-router-dom";
import IntroPanel from "./IntroPanel";
import CodePanelDesign from "./CodePanelDesign";

function HeroSection() {
  return (
    <div className="bg-gradient-to-r from-blue-600 via-purple-600 to-indigo-600 text-white">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-20 grid md:grid-cols-2 gap-12 items-center">
        <IntroPanel />
        <CodePanelDesign />
      </div>
    </div>
  );
}

export default HeroSection;
