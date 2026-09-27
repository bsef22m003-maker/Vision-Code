function LearningStep({
  number,
  bgColor,
  title,
  description,
  delay = "delay-100",
}) {
  return (
    <div className={`text-center scroll-fade-up ${delay}`}>
      <div
        className={`w-16 h-16 ${bgColor} text-white rounded-full flex items-center justify-center text-2xl font-bold mx-auto mb-4`}
      >
        {number}
      </div>
      <h3 className="text-lg font-bold text-gray-900 mb-2">{title}</h3>
      <p className="text-gray-600 text-sm">{description}</p>
    </div>
  );
}

export default LearningStep;
