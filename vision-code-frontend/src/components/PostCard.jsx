import { useState } from "react";

function PostCard({ post }) {
  const [votes, setVotes] = useState(post.votes);
  const [userVote, setUserVote] = useState(null); // null, 'up', or 'down'

  const handleUpvote = () => {
    if (userVote === "up") {
      setVotes(votes - 1);
      setUserVote(null);
    } else if (userVote === "down") {
      setVotes(votes + 2);
      setUserVote("up");
    } else {
      setVotes(votes + 1);
      setUserVote("up");
    }
  };

  const handleDownvote = () => {
    if (userVote === "down") {
      setVotes(votes + 1);
      setUserVote(null);
    } else if (userVote === "up") {
      setVotes(votes - 2);
      setUserVote("down");
    } else {
      setVotes(votes - 1);
      setUserVote("down");
    }
  };

  const typeColors = {
    yellow: "bg-yellow-100 text-yellow-800",
    green: "bg-green-100 text-green-800",
    blue: "bg-blue-100 text-blue-800",
    purple: "bg-purple-100 text-purple-800",
  };

  return (
    <div className="bg-white p-6 rounded-xl shadow-sm border border-gray-100 hover:border-blue-200 transition cursor-pointer">
      <div className="flex gap-4">
        {/* Vote Section */}
        <div className="flex flex-col items-center gap-1 text-gray-500">
          <button
            onClick={handleUpvote}
            className={`hover:text-blue-600 transition ${
              userVote === "up" ? "text-blue-600" : ""
            }`}
          >
            <svg
              className="w-6 h-6"
              fill="none"
              stroke="currentColor"
              viewBox="0 0 24 24"
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth="2"
                d="M5 15l7-7 7 7"
              ></path>
            </svg>
          </button>
          <span className="font-bold text-sm">{votes}</span>
          <button
            onClick={handleDownvote}
            className={`hover:text-red-600 transition ${
              userVote === "down" ? "text-red-600" : ""
            }`}
          >
            <svg
              className="w-6 h-6"
              fill="none"
              stroke="currentColor"
              viewBox="0 0 24 24"
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth="2"
                d="M19 9l-7 7-7-7"
              ></path>
            </svg>
          </button>
        </div>

        {/* Content Section */}
        <div className="flex-1">
          <div className="flex items-center gap-2 mb-2">
            <span
              className={`${
                typeColors[post.typeColor]
              } text-xs px-2 py-0.5 rounded-full font-medium`}
            >
              {post.type}
            </span>
            <span className="text-xs text-gray-500">
              Posted by {post.author} • {post.timeAgo}
            </span>
          </div>

          <h3 className="text-lg font-bold text-gray-900 mb-2">{post.title}</h3>
          <p className="text-gray-600 text-sm mb-4">{post.content}</p>

          {/* Image Placeholder */}
          {post.hasImage && (
            <div className="h-32 bg-gray-200 rounded-lg mb-4 flex items-center justify-center text-gray-400">
              [Project Screenshot Placeholder]
            </div>
          )}

          {/* Post Stats */}
          <div className="flex gap-4 text-sm text-gray-500">
            <span className="flex items-center gap-1">
              <svg
                className="w-4 h-4"
                fill="none"
                stroke="currentColor"
                viewBox="0 0 24 24"
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth="2"
                  d="M8 12h.01M12 12h.01M16 12h.01M21 12c0 4.418-4.03 8-9 8a9.863 9.863 0 01-4.255-.949L3 20l1.395-3.72C3.512 15.042 3 13.574 3 12c0-4.418 4.03-8 9-8s9 3.582 9 8z"
                ></path>
              </svg>
              {post.comments} comments
            </span>
            {post.views && (
              <span className="flex items-center gap-1">
                <svg
                  className="w-4 h-4"
                  fill="none"
                  stroke="currentColor"
                  viewBox="0 0 24 24"
                >
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    strokeWidth="2"
                    d="M15 12a3 3 0 11-6 0 3 3 0 016 0z"
                  ></path>
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    strokeWidth="2"
                    d="M2.458 12C3.732 7.943 7.523 5 12 5c4.478 0 8.268 2.943 9.542 7-1.274 4.057-5.064 7-9.542 7-4.477 0-8.268-2.943-9.542-7z"
                  ></path>
                </svg>
                {post.views} views
              </span>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}

export default PostCard;
