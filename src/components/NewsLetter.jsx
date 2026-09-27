import { useState } from "react";
import Button from "./Button";

function NewsletterCTA() {
  const [email, setEmail] = useState("");

  const handleSubscribe = (e) => {
    e.preventDefault();

    // Add subscription logic here
    setEmail("");
  };

  return (
    <div className="max-w-7xl mx-auto px-4 sm:px-6 lg: px-8 pb-20">
      <div className="bg-gradient-to-r from-blue-600 via-purple-600 to-indigo-600 rounded-2xl p-12 text-center text-white relative overflow-hidden">
        <div className="absolute top-0 right-0 w-64 h-64 bg-white/10 rounded-full blur-3xl"></div>
        <div className="absolute bottom-0 left-0 w-64 h-64 bg-purple-500/20 rounded-full blur-3xl"></div>
        <div className="relative z-10">
          <h2 className="text-3xl font-bold mb-4">
            Stay Updated with New Courses
          </h2>
          <p className="text-blue-100 mb-6 max-w-xl mx-auto">
            Get notified when we launch new courses and exclusive learning
            opportunities
          </p>
          <form
            onSubmit={handleSubscribe}
            className="flex gap-3 max-w-md mx-auto"
          >
            <input
              type="email"
              value={email}
              onChange={(e) => setEmail(e.target.value)}
              placeholder="Enter your email"
              required
              className="flex-1 px-6 py-3 rounded-lg text-gray-800 border-0 focus:outline-none focus: ring-2 focus:ring-white"
            />
            <Button
              type="submit"
              className="px-6 py-3  text-blue-600 rounded-lg hover:shadow-lg font-semibold"
            >
              Subscribe
            </Button>
          </form>
        </div>
      </div>
    </div>
  );
}

export default NewsletterCTA;
