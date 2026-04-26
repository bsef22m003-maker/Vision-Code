import React from "react";

function CodePanelDesign() {
  return (
    <div className="hidden md:block transform hover:scale-105 transition duration-500 scroll-slide-right">
      <div className="bg-gray-900/80 backdrop-blur-xl rounded-2xl p-6 border border-white/20 shadow-2xl">
        <div className="flex justify-between items-center mb-4 border-b border-gray-700 pb-2">
          <div className="flex space-x-2">
            <div className="w-3 h-3 bg-red-500 rounded-full"></div>
            <div className="w-3 h-3 bg-yellow-500 rounded-full"></div>
            <div className="w-3 h-3 bg-green-500 rounded-full"></div>
          </div>
          <span className="text-xs text-gray-400 font-mono">script.js</span>
        </div>

        <div className="font-mono text-sm leading-relaxed">
          <p className="text-pink-400">
            const <span className="text-blue-400">user</span> = {"{"}
          </p>
          <p className="text-green-400 pl-4">
            name: <span className="text-yellow-300">'Developer'</span>,
          </p>
          <p className="text-green-400 pl-4">
            level: <span className="text-purple-400">99</span>,
          </p>
          <p className="text-blue-400 pl-4">
            startLearning: <span className="text-pink-400">function</span>(){" "}
            {"{"}
          </p>
          <p className="text-gray-300 pl-8">
            return <span className="text-yellow-300">'Future Unlocked'</span>;
          </p>
          <p className="text-blue-400 pl-4">{"}"}</p>
          <p className="text-pink-400">{"}"};</p>
        </div>
      </div>
    </div>
  );
}

export default CodePanelDesign;
