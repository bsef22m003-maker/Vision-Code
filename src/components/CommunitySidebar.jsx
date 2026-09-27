function CommunitySidebar() {
  const trendingTopics = [
    "#javascript",
    "#help",
    "#showcase",
    "#react",
    "#career",
  ];

  const topContributors = [
    { name: "John Doe", initials: "JD", points: "1.2k", color: "blue" },
    { name: "Alice Smith", initials: "AS", points: "980", color: "purple" },
  ];

  return (
    <div className="lg:col-span-1 space-y-6">
      <div className="bg-white p-4 rounded-xl shadow-sm border border-gray-100">
        <h3 className="font-bold text-gray-900 mb-4">Trending Topics</h3>
        <div className="flex flex-wrap gap-2">
          {trendingTopics.map((topic) => (
            <span
              key={topic}
              className="px-3 py-1 bg-gray-100 text-xs rounded-full hover: bg-gray-200 cursor-pointer transition"
            >
              {topic}
            </span>
          ))}
        </div>
      </div>

      <div className="bg-white p-4 rounded-xl shadow-sm border border-gray-100">
        <h3 className="font-bold text-gray-900 mb-4">Top Contributors</h3>
        <ul className="space-y-3">
          {topContributors.map((contributor) => (
            <li key={contributor.initials} className="flex items-center gap-3">
              <div
                className={`w-8 h-8 rounded-full bg-${contributor.color}-100 text-${contributor.color}-600 flex items-center justify-center font-bold text-xs`}
              >
                {contributor.initials}
              </div>
              <div className="text-sm">
                <p className="font-medium">{contributor.name}</p>
                <p className="text-xs text-gray-500">
                  {contributor.points} pts
                </p>
              </div>
            </li>
          ))}
        </ul>
      </div>
    </div>
  );
}

export default CommunitySidebar;
