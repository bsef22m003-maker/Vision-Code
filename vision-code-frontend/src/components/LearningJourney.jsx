import LearningStep from "./LearningStep";

function LearningJourney() {
  const learningSteps = [
    {
      number: 1,
      bgColor: "bg-blue-600",
      title: "Choose Your Path",
      description:
        "Select from web development, data science, or mobile development.",
      delay: "delay-100",
    },
    {
      number: 2,
      bgColor: "bg-purple-600",
      title: "Learn & Practice",
      description: "Complete interactive lessons and build real projects.",
      delay: "delay-200",
    },
    {
      number: 3,
      bgColor: "bg-indigo-600",
      title: "Get Feedback",
      description: "Receive guidance from community and automated reviews.",
      delay: "delay-300",
    },
    {
      number: 4,
      bgColor: "bg-green-600",
      title: "Earn Certificate",
      description: "Complete courses and showcase your achievements.",
      delay: "delay-400",
    },
  ];

  return (
    <div className="py-20 bg-gradient-to-br from-blue-50 to-purple-50">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="text-center mb-16 scroll-fade-up">
          <h2 className="text-4xl font-bold text-gray-900 mb-4">
            Your Learning Journey
          </h2>
          <p className="text-xl text-gray-600">
            Follow a proven path to success
          </p>
        </div>

        <div className="grid md:grid-cols-4 gap-8">
          {learningSteps.map((step, index) => (
            <LearningStep
              key={index}
              number={step.number}
              bgColor={step.bgColor}
              title={step.title}
              description={step.description}
              delay={step.delay}
            />
          ))}
        </div>
      </div>
    </div>
  );
}

export default LearningJourney;
