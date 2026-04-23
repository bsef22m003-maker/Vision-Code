import { useState } from "react";
import { NavLink, useNavigate } from "react-router-dom";
import NavBar from "../components/NavBar";
import Footer from "../components/Footer";
import Button from "../components/Button";

function Practice() {
  const navigate = useNavigate();
  const [selectedLanguage, setSelectedLanguage] = useState("javascript");

  const handleLaunchEditor = () => {
    navigate("/practice/editor");
  };

  const codeExamples = {
    javascript: `console.log("Hello World!");
// Start typing your code here...`,
    python: `print("Hello World!")
# Start typing your code here...`,
    java: `public class Main {
    public static void main(String[] args) {
        System.out.println("Hello World!");
        // Start typing your code here... 
    }
}`,
    cpp: `#include <iostream>
using namespace std;

int main() {
    cout << "Hello World!" << endl;
    // Start typing your code here...
    return 0;
}`,
  };

  return (
    <div className="min-h-screen flex flex-col bg-gray-50">
      {/* NavBar */}
      <NavBar
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
          className="text-gray-700 hover:text-blue-600 font-medium transition"
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
          className="text-blue-600 font-medium border-b-2 border-blue-600 pb-1"
        >
          Practice
        </NavLink>
      </NavBar>

      {/* Main Content */}
      <div className="flex-1">
        <div className="max-w-4xl mx-auto py-20 px-4 text-center">
          {/* Header */}
          <div className="mb-8">
            <div className="inline-block p-4 bg-blue-100 rounded-full mb-4">
              <svg
                className="w-12 h-12 text-blue-500"
                fill="none"
                stroke="currentColor"
                viewBox="0 0 24 24"
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth="2"
                  d="M10 20l4-16m4 4l4 4-4 4M6 16l-4-4 4-4"
                />
              </svg>
            </div>
            <h2 className="text-4xl font-bold mb-4 text-gray-900">
              Coding Playground
            </h2>
            <p className="text-xl text-gray-600">
              Practice your skills with our online compiler.
            </p>
          </div>

          {/* Language Selector */}
          <div className="mb-6 flex justify-center gap-3 flex-wrap">
            <button
              onClick={() => setSelectedLanguage("javascript")}
              className={`px-4 py-2 rounded-lg font-medium text-sm transition ${
                selectedLanguage === "javascript"
                  ? "bg-yellow-500 text-white shadow-lg"
                  : "bg-white text-gray-700 border border-gray-300 hover:bg-gray-50"
              }`}
            >
              JavaScript
            </button>
            <button
              onClick={() => setSelectedLanguage("python")}
              className={`px-4 py-2 rounded-lg font-medium text-sm transition ${
                selectedLanguage === "python"
                  ? "bg-blue-500 text-white shadow-lg"
                  : "bg-white text-gray-700 border border-gray-300 hover:bg-gray-50"
              }`}
            >
              Python
            </button>
            <button
              onClick={() => setSelectedLanguage("java")}
              className={`px-4 py-2 rounded-lg font-medium text-sm transition ${
                selectedLanguage === "java"
                  ? "bg-red-500 text-white shadow-lg"
                  : "bg-white text-gray-700 border border-gray-300 hover:bg-gray-50"
              }`}
            >
              Java
            </button>
            <button
              onClick={() => setSelectedLanguage("cpp")}
              className={`px-4 py-2 rounded-lg font-medium text-sm transition ${
                selectedLanguage === "cpp"
                  ? "bg-purple-500 text-white shadow-lg"
                  : "bg-white text-gray-700 border border-gray-300 hover:bg-gray-50"
              }`}
            >
              C++
            </button>
          </div>

          {/* Code Editor Preview */}
          <div className="bg-gray-900 rounded-lg shadow-2xl overflow-hidden text-left font-mono text-sm">
            {/* Terminal Header */}
            <div className="bg-gray-800 p-3 flex items-center justify-between">
              <div className="flex space-x-2">
                <div className="w-3 h-3 bg-red-500 rounded-full"></div>
                <div className="w-3 h-3 bg-yellow-500 rounded-full"></div>
                <div className="w-3 h-3 bg-green-500 rounded-full"></div>
              </div>
              <span className="text-gray-400 text-xs uppercase">
                {selectedLanguage}
              </span>
            </div>

            {/* Code Area */}
            <div className="p-6 text-white min-h-[16rem] relative">
              <pre className="whitespace-pre-wrap">
                <code>{codeExamples[selectedLanguage]}</code>
              </pre>

              {/* Typing Cursor */}
              <div className="absolute bottom-6 left-6 w-2 h-5 bg-white animate-pulse"></div>
            </div>
          </div>

          {/* Action Buttons */}
          <div className="mt-8 flex justify-center gap-4 flex-wrap">
            <Button
              onClick={handleLaunchEditor}
              className="px-8 py-3 bg-gradient-to-r from-blue-600 to-purple-600 text-white rounded-lg hover: shadow-xl transform hover:scale-105 transition-all font-semibold"
            >
              Launch Full Editor
            </Button>
            <button
              onClick={() => navigate("/challenges")}
              className="px-8 py-3 bg-white text-gray-700 border-2 border-gray-300 rounded-lg hover:bg-gray-50 hover:border-blue-500 transition font-semibold"
            >
              Browse Challenges
            </button>
          </div>

          {/* Features Grid */}
          <div className="mt-20 grid grid-cols-1 md:grid-cols-3 gap-6 text-left">
            <div className="bg-white p-6 rounded-xl shadow-md border border-gray-100 hover:shadow-lg transition">
              <div className="w-12 h-12 bg-blue-100 rounded-lg flex items-center justify-center mb-4">
                <svg
                  className="w-6 h-6 text-blue-600"
                  fill="none"
                  stroke="currentColor"
                  viewBox="0 0 24 24"
                >
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    strokeWidth="2"
                    d="M13 10V3L4 14h7v7l9-11h-7z"
                  />
                </svg>
              </div>
              <h3 className="text-xl font-bold text-gray-900 mb-2">
                Instant Execution
              </h3>
              <p className="text-gray-600">
                Run your code instantly without any setup. Support for multiple
                languages.
              </p>
            </div>

            <div className="bg-white p-6 rounded-xl shadow-md border border-gray-100 hover:shadow-lg transition">
              <div className="w-12 h-12 bg-green-100 rounded-lg flex items-center justify-center mb-4">
                <svg
                  className="w-6 h-6 text-green-600"
                  fill="none"
                  stroke="currentColor"
                  viewBox="0 0 24 24"
                >
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    strokeWidth="2"
                    d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z"
                  />
                </svg>
              </div>
              <h3 className="text-xl font-bold text-gray-900 mb-2">
                Smart Suggestions
              </h3>
              <p className="text-gray-600">
                Get intelligent code completion and syntax highlighting as you
                type.
              </p>
            </div>

            <div className="bg-white p-6 rounded-xl shadow-md border border-gray-100 hover:shadow-lg transition">
              <div className="w-12 h-12 bg-purple-100 rounded-lg flex items-center justify-center mb-4">
                <svg
                  className="w-6 h-6 text-purple-600"
                  fill="none"
                  stroke="currentColor"
                  viewBox="0 0 24 24"
                >
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    strokeWidth="2"
                    d="M8 7V3m8 4V3m-9 8h10M5 21h14a2 2 0 002-2V7a2 2 0 00-2-2H5a2 2 0 00-2 2v12a2 2 0 002 2z"
                  />
                </svg>
              </div>
              <h3 className="text-xl font-bold text-gray-900 mb-2">
                Save & Share
              </h3>
              <p className="text-gray-600">
                Save your code snippets and share them with the community.
              </p>
            </div>
          </div>
        </div>
      </div>

      {/* Footer */}
      <Footer />
    </div>
  );
}

export default Practice;
