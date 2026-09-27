function Footer() {
  return (
    <footer className="bg-gray-900 text-gray-400 py-12">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 grid grid-cols-1 md:grid-cols-4 gap-8">
        <div className="col-span-1 md:col-span-2">
          <span className="text-2xl font-bold text-white mb-4 block">
            Vision-Code
          </span>
          <p className="mb-4">
            The best place to start your coding journey. Interactive,
            structured, and community-driven.
          </p>
        </div>
        <div>
          <h4 className="text-white font-bold mb-4">Platform</h4>
          <ul className="space-y-2 text-sm">
            <li>
              <a href="courses.html" className="hover:text-white">
                Browse Courses
              </a>
            </li>
            <li>
              <a href="practice.html" className="hover:text-white">
                Practice
              </a>
            </li>
            <li>
              <a href="community.html" className="hover:text-white">
                Community
              </a>
            </li>
          </ul>
        </div>
        <div>
          <h4 className="text-white font-bold mb-4">Company</h4>
          <ul className="space-y-2 text-sm">
            <li>
              <a href="#" className="hover:text-white">
                About Us
              </a>
            </li>
            <li>
              <a href="#" className="hover:text-white">
                Privacy Policy
              </a>
            </li>
            <li>
              <a href="#" className="hover:text-white">
                Terms of Service
              </a>
            </li>
          </ul>
        </div>
      </div>
    </footer>
  );
}

export default Footer;
