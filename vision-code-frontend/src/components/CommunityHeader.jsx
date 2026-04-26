function CommunityHeader() {
  const handleNewPost = () => {
    console.log("Create new post");
  };

  return (
    <div className="bg-white  sticky top-16 z-40">
      <div className="max-w-7xl mx-auto px-4 py-4 flex justify-between items-center">
        <h2 className="text-2xl font-bold text-gray-800">
          Developer Community
        </h2>
        <button
          onClick={handleNewPost}
          className="bg-blue-600 text-white px-4 py-2 rounded-lg text-sm font-medium hover: bg-blue-700 transition flex items-center gap-2"
        >
          <span>+</span> New Post
        </button>
      </div>
    </div>
  );
}

export default CommunityHeader;
