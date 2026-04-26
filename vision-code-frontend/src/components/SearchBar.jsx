function SearchBar() {
  const handleSearch = (e) => {
    e.preventDefault();
    // Add search logic here
  };

  return (
    <div className="max-w-2xl mx-auto relative">
      <div className="relative group">
        <div className="absolute -inset-0.5 bg-gradient-to-r from-pink-600 via-purple-600 to-blue-600 rounded-full opacity-50 group-hover:opacity-75 blur transition duration-500"></div>
        <div className="relative bg-white rounded-full shadow-2xl">
          <input
            type="text"
            placeholder="Search for Python, React, Java, Machine Learning..."
            className="w-full px-8 py-4 rounded-full text-gray-800 focus:outline-none focus:ring-2 focus:ring-blue-500 text-lg"
          />
          <button
            onClick={handleSearch}
            className="absolute right-3 top-3 bg-gradient-to-r from-blue-600 to-purple-600 p-2.5 rounded-full hover:shadow-lg transition"
          >
            <svg
              xmlns="http://www.w3.org/2000/svg"
              className="h-6 w-6 text-white"
              fill="none"
              viewBox="0 0 24 24"
              stroke="currentColor"
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth="2"
                d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z"
              />
            </svg>
          </button>
        </div>
      </div>
    </div>
  );
}

export default SearchBar;
