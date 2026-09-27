import React from "react";
import { NavLink } from "react-router-dom";

export default function Navbar({ appName, children, authButtons }) {
  return (
    <nav className="bg-white shadow-sm sticky top-0 z-50">
      <div className="max-w-7xl mx-auto px-4 sm: px-6 lg:px-8">
        <div className="flex justify-between items-center h-16">
          {/* Left side - Logo and all children (nav links) */}
          <div className="flex items-center space-x-8">
            <NavLink to="/" className="flex items-center space-x-2">
              <div className="bg-gradient-to-r from-blue-600 to-purple-600 p-2 rounded-lg">
                <svg
                  xmlns="http://www.w3.org/2000/svg"
                  width="24"
                  height="24"
                  fill="none"
                  stroke="white"
                  strokeWidth="2"
                >
                  <polyline points="16 18 22 12 16 6"></polyline>
                  <polyline points="8 6 2 12 8 18"></polyline>
                </svg>
              </div>
              <span className="text-xl font-bold gradient-text">{appName}</span>
            </NavLink>

            {/* All children (navigation links) go here */}
            <div className="hidden md:flex space-x-6">{children}</div>
          </div>

          {/* Right side - Auth buttons */}
          <div className="flex items-center space-x-4">{authButtons}</div>
        </div>
      </div>
    </nav>
  );
}
