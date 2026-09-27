import PostCard from "./PostCard";

const postsData = [
  {
    id: 1,
    type: "Question",
    typeColor: "yellow",
    author: "@sarah_dev",
    timeAgo: "2h ago",
    title: "How do I center a div perfectly?",
    content:
      "I've tried flexbox and grid but sometimes it acts weird on mobile Safari.  Any robust snippets?",
    votes: 42,
    comments: 15,
    views: 120,
    hasImage: false,
  },
  {
    id: 2,
    type: "Showcase",
    typeColor: "green",
    author: "@mike_codes",
    timeAgo: "5h ago",
    title: "Built my first Portfolio using Tailwind!  🚀",
    content:
      "Check it out and let me know what you think. I used the CodeLearn grid module for the layout.",
    votes: 128,
    comments: 42,
    views: null,
    hasImage: true,
  },
];

function CommunityFeed() {
  return (
    <div className="lg:col-span-3 space-y-4">
      {postsData.map((post) => (
        <PostCard key={post.id} post={post} />
      ))}
    </div>
  );
}

export default CommunityFeed;
