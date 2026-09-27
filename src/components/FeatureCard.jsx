function FeatureCard({
  icon,
  title,
  description,
  delay = "delay-100",
  bgColor = "bg-blue-100",
  iconColor = "text-blue-600",
}) {
  return (
    <div className={`relative group scroll-fade-up ${delay}`}>
      {/* Animated gradient background */}
      <div className="absolute -inset-0.5 bg-gradient-to-r from-pink-600 via-purple-600 to-blue-600 rounded-xl opacity-0 group-hover:opacity-100 blur-lg transition-all duration-500 group-hover:duration-200 animate-gradient"></div>

      {/* Card content */}
      <div className="relative bg-white p-8 rounded-xl shadow-md hover:shadow-xl transition-all duration-300">
        <div
          className={`w-12 h-12 ${bgColor} rounded-lg flex items-center justify-center mb-4`}
        >
          <div className={`w-6 h-6 ${iconColor}`}>{icon}</div>
        </div>
        <h3 className="text-xl font-bold text-gray-900 mb-2">{title}</h3>
        <p className="text-gray-600">{description}</p>
      </div>
    </div>
  );
}

export default FeatureCard;
