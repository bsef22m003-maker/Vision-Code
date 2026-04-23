import { useState, useEffect } from "react";
import Button from "./Button";

function QuizModal({ quiz, onClose }) {
  const [currentQuestion, setCurrentQuestion] = useState(0);
  const [selectedAnswers, setSelectedAnswers] = useState({});
  const [showResults, setShowResults] = useState(false);
  const [score, setScore] = useState(0);
  const [secondsRemaining, setSecondsRemaining] = useState(null);
  const [isTimerActive, setIsTimerActive] = useState(false);

  const questions = quiz.parsedContent.questions;
  const totalTime = (quiz.estimated_duration_minutes || 15) * 60;

  useEffect(() => {
    if (secondsRemaining === null) {
      setSecondsRemaining(totalTime);
      setIsTimerActive(true);
    }
  }, [totalTime]);

  useEffect(() => {
    if (!isTimerActive || secondsRemaining === null) return;

    const timerId = setInterval(() => {
      setSecondsRemaining((prev) => {
        if (prev <= 1) {
          setIsTimerActive(false);
          calculateScore();
          return 0;
        }
        return prev - 1;
      });
    }, 1000);

    return () => clearInterval(timerId);
  }, [isTimerActive, secondsRemaining]);

  const formatTime = (seconds) => {
    const mins = Math.floor(seconds / 60);
    const secs = seconds % 60;
    return `${mins.toString().padStart(2, "0")}:${secs
      .toString()
      .padStart(2, "0")}`;
  };

  const handleAnswerSelect = (answerIndex) => {
    setSelectedAnswers({
      ...selectedAnswers,
      [currentQuestion]: answerIndex,
    });
  };

  const handleNext = () => {
    if (currentQuestion < questions.length - 1) {
      setCurrentQuestion(currentQuestion + 1);
    } else {
      finishQuiz();
    }
  };

  const handlePrevious = () => {
    if (currentQuestion > 0) {
      setCurrentQuestion(currentQuestion - 1);
    }
  };

  const finishQuiz = () => {
    setIsTimerActive(false);
    calculateScore();
  };

  const calculateScore = () => {
    let correct = 0;
    questions.forEach((question, index) => {
      if (selectedAnswers[index] === question.correct_answer) {
        correct++;
      }
    });
    setScore(correct);
    setShowResults(true);
  };

  const handleRetake = () => {
    setCurrentQuestion(0);
    setSelectedAnswers({});
    setShowResults(false);
    setScore(0);
    setSecondsRemaining(totalTime);
    setIsTimerActive(true);
  };

  const percentage = Math.round((score / questions.length) * 100);
  const passed = percentage >= (quiz.metadata?.passing_score || 70);

  const answeredQuestions = Object.keys(selectedAnswers).length;
  const progressPercentage = ((currentQuestion + 1) / questions.length) * 100;

  const getTimerColor = () => {
    if (secondsRemaining === null) return "text-gray-600";
    const timePercentage = (secondsRemaining / totalTime) * 100;
    if (timePercentage > 50) return "text-green-600";
    if (timePercentage > 25) return "text-yellow-600";
    return "text-red-600";
  };

  const getTimerBgGradient = () => {
    if (secondsRemaining === null) return "from-gray-100 to-gray-50";
    const timePercentage = (secondsRemaining / totalTime) * 100;
    if (timePercentage > 50) return "from-green-50 to-emerald-50";
    if (timePercentage > 25) return "from-yellow-50 to-orange-50";
    return "from-red-50 to-pink-50";
  };

  if (showResults) {
    const timeTaken = totalTime - (secondsRemaining || 0);

    return (
      <div className="fixed inset-0 bg-black/60 backdrop-blur-md z-50 flex items-center justify-center p-4 animate-fadeIn">
        <div className="bg-white rounded-3xl shadow-2xl w-full max-w-2xl animate-slideUp">
          {/* Decorative Header - Reduced */}
          <div
            className={`relative h-20 ${
              passed
                ? "bg-gradient-to-br from-green-400 via-emerald-500 to-teal-600"
                : "bg-gradient-to-br from-orange-400 via-red-500 to-pink-600"
            } rounded-t-3xl overflow-hidden`}
          >
            <div className="absolute inset-0 opacity-20">
              <div className="absolute top-0 left-0 w-32 h-32 bg-white rounded-full -translate-x-1/2 -translate-y-1/2"></div>
              <div className="absolute bottom-0 right-0 w-32 h-32 bg-white rounded-full translate-x-1/2 translate-y-1/2"></div>
            </div>
            {/* Close Button */}
            <button
              onClick={onClose}
              className="absolute top-4 right-4 w-8 h-8 bg-white/20 hover:bg-white/30 rounded-lg flex items-center justify-center transition-all hover:rotate-90 duration-300 z-10"
            >
              <svg
                className="w-5 h-5 text-white"
                fill="none"
                stroke="currentColor"
                viewBox="0 0 24 24"
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth="2"
                  d="M6 18L18 6M6 6l12 12"
                />
              </svg>
            </button>
          </div>

          <div className="px-8 py-6 -mt-10 relative">
            {/* Score Circle - Smaller */}
            <div className="relative mx-auto w-20 h-20 mb-4">
              <svg className="w-20 h-20 transform -rotate-90">
                <circle
                  cx="40"
                  cy="40"
                  r="36"
                  stroke="currentColor"
                  strokeWidth="5"
                  fill="white"
                  className="text-gray-200"
                />
                <circle
                  cx="40"
                  cy="40"
                  r="36"
                  stroke="currentColor"
                  strokeWidth="5"
                  fill="none"
                  className={passed ? "text-green-500" : "text-red-500"}
                  strokeDasharray={`${2 * Math.PI * 36}`}
                  strokeDashoffset={`${
                    2 * Math.PI * 36 * (1 - percentage / 100)
                  }`}
                  strokeLinecap="round"
                  style={{ transition: "stroke-dashoffset 1s ease-in-out" }}
                />
              </svg>
              <div className="absolute inset-0 flex items-center justify-center">
                <div className="text-center">
                  <p className="text-2xl font-bold text-gray-900">
                    {percentage}
                    <span className="text-sm">%</span>
                  </p>
                </div>
              </div>
            </div>

            {/* Result Message - Compact */}
            <div className="text-center mb-5">
              <h2 className="text-2xl font-bold text-gray-900 mb-1">
                {passed ? "Outstanding!   🎉" : "Keep Practicing!  💪"}
              </h2>
              <p className="text-gray-600 text-sm">
                You answered{" "}
                <span className="font-bold text-gray-900">{score}</span> out of{" "}
                <span className="font-bold text-gray-900">
                  {questions.length}
                </span>{" "}
                questions correctly
              </p>
            </div>

            {/* Stats Grid - More Compact */}
            <div className="grid grid-cols-4 gap-3 mb-5">
              <div className="bg-gradient-to-br from-blue-50 to-blue-100 border border-blue-200 rounded-xl p-3 text-center">
                <div className="w-8 h-8 bg-blue-500 rounded-lg flex items-center justify-center mx-auto mb-1. 5">
                  <svg
                    className="w-4 h-4 text-white"
                    fill="none"
                    stroke="currentColor"
                    viewBox="0 0 24 24"
                  >
                    <path
                      strokeLinecap="round"
                      strokeLinejoin="round"
                      strokeWidth="2"
                      d="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z"
                    />
                  </svg>
                </div>
                <p className="text-lg font-bold text-blue-900">
                  {formatTime(timeTaken)}
                </p>
                <p className="text-xs text-blue-700 font-medium mt-0.5">
                  Time Taken
                </p>
              </div>

              <div className="bg-gradient-to-br from-purple-50 to-purple-100 border border-purple-200 rounded-xl p-3 text-center">
                <div className="w-8 h-8 bg-purple-500 rounded-lg flex items-center justify-center mx-auto mb-1.5">
                  <svg
                    className="w-4 h-4 text-white"
                    fill="none"
                    stroke="currentColor"
                    viewBox="0 0 24 24"
                  >
                    <path
                      strokeLinecap="round"
                      strokeLinejoin="round"
                      strokeWidth="2"
                      d="M13 10V3L4 14h7v7l9-11h-7z"
                    />
                  </svg>
                </div>
                <p className="text-lg font-bold text-purple-900">
                  {formatTime(secondsRemaining || 0)}
                </p>
                <p className="text-xs text-purple-700 font-medium mt-0.5">
                  Time Saved
                </p>
              </div>

              <div className="bg-gradient-to-br from-green-50 to-green-100 border border-green-200 rounded-xl p-3 text-center">
                <div className="w-8 h-8 bg-green-500 rounded-lg flex items-center justify-center mx-auto mb-1.5">
                  <svg
                    className="w-4 h-4 text-white"
                    fill="none"
                    stroke="currentColor"
                    viewBox="0 0 24 24"
                  >
                    <path
                      strokeLinecap="round"
                      strokeLinejoin="round"
                      strokeWidth="2"
                      d="M5 13l4 4L19 7"
                    />
                  </svg>
                </div>
                <p className="text-lg font-bold text-green-900">{score}</p>
                <p className="text-xs text-green-700 font-medium mt-0.5">
                  Correct
                </p>
              </div>

              <div className="bg-gradient-to-br from-red-50 to-red-100 border border-red-200 rounded-xl p-3 text-center">
                <div className="w-8 h-8 bg-red-500 rounded-lg flex items-center justify-center mx-auto mb-1.5">
                  <svg
                    className="w-4 h-4 text-white"
                    fill="none"
                    stroke="currentColor"
                    viewBox="0 0 24 24"
                  >
                    <path
                      strokeLinecap="round"
                      strokeLinejoin="round"
                      strokeWidth="2"
                      d="M6 18L18 6M6 6l12 12"
                    />
                  </svg>
                </div>
                <p className="text-lg font-bold text-red-900">
                  {questions.length - score}
                </p>
                <p className="text-xs text-red-700 font-medium mt-0.5">
                  Incorrect
                </p>
              </div>
            </div>

            {/* Pass/Fail Badge - Inline */}
            <div
              className={`${
                passed
                  ? "bg-gradient-to-r from-green-500 to-emerald-500"
                  : "bg-gradient-to-r from-orange-500 to-red-500"
              } text-white rounded-xl p-3.5 mb-5 shadow-lg`}
            >
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-2">
                  <div className="w-8 h-8 bg-white/20 rounded-lg flex items-center justify-center">
                    {passed ? (
                      <svg
                        className="w-5 h-5"
                        fill="none"
                        stroke="currentColor"
                        viewBox="0 0 24 24"
                      >
                        <path
                          strokeLinecap="round"
                          strokeLinejoin="round"
                          strokeWidth="2"
                          d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z"
                        />
                      </svg>
                    ) : (
                      <svg
                        className="w-5 h-5"
                        fill="none"
                        stroke="currentColor"
                        viewBox="0 0 24 24"
                      >
                        <path
                          strokeLinecap="round"
                          strokeLinejoin="round"
                          strokeWidth="2"
                          d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z"
                        />
                      </svg>
                    )}
                  </div>
                  <div>
                    <p className="font-bold text-sm">
                      {passed
                        ? "Congratulations!  You Passed ✓"
                        : "Not Passed - Keep Learning"}
                    </p>
                    <p className="text-xs opacity-90">
                      Required: {quiz.metadata?.passing_score || 70}% • Your
                      Score: {percentage}%
                    </p>
                  </div>
                </div>
                <div className="text-right">
                  <div
                    className={`text-2xl font-bold ${
                      passed ? "text-white" : "text-white"
                    }`}
                  >
                    {passed ? "🎉" : "📚"}
                  </div>
                </div>
              </div>
            </div>

            {/* Action Buttons - Side by Side */}
            <div className="grid grid-cols-2 gap-3">
              <Button
                onClick={handleRetake}
                variant="outline"
                className="w-full py-3 font-semibold border-2 hover:bg-gray-50 flex items-center justify-center gap-2"
              >
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
                    d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15"
                  />
                </svg>
                Retake Quiz
              </Button>
              <Button
                onClick={onClose}
                className="w-full bg-gradient-to-r from-blue-600 to-purple-600 text-white py-3 font-semibold shadow-lg hover:shadow-xl transform hover:scale-[1.02] transition-all flex items-center justify-center gap-2"
              >
                Continue
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
                    d="M13 7l5 5m0 0l-5 5m5-5H6"
                  />
                </svg>
              </Button>
            </div>
          </div>
        </div>
      </div>
    );
  }

  const currentQ = questions[currentQuestion];

  return (
    <div className="fixed inset-0 bg-black/60 backdrop-blur-md z-50 flex items-center justify-center p-2 sm:p-4">
      <div className="bg-white rounded-3xl shadow-2xl w-full max-w-4xl h-[95vh] flex flex-col overflow-hidden">
        {/* Fixed Header - More Compact */}
        <div className="relative bg-gradient-to-r from-green-500 via-emerald-500 to-teal-600 text-white px-4 sm:px-6 py-2. 5 flex-shrink-0">
          {/* Decorative Background */}
          <div className="absolute inset-0 opacity-10">
            <div className="absolute top-0 right-0 w-64 h-64 bg-white rounded-full -translate-y-1/2 translate-x-1/2"></div>
            <div className="absolute bottom-0 left-0 w-48 h-48 bg-white rounded-full translate-y-1/2 -translate-x-1/2"></div>
          </div>

          <div className="relative z-10">
            {/* Title and Close */}
            <div className="flex justify-between items-start mb-2">
              <div className="flex-1">
                <h2 className="text-lg sm:text-xl font-bold mb-0.5">
                  {quiz.title}
                </h2>
                <p className="text-green-100 text-xs">
                  Question {currentQuestion + 1} of {questions.length}
                </p>
              </div>
              <button
                onClick={onClose}
                className="w-8 h-8 bg-white/20 hover:bg-white/30 rounded-xl flex items-center justify-center transition-all hover:rotate-90 duration-300 flex-shrink-0 ml-2"
              >
                <svg
                  className="w-5 h-5"
                  fill="none"
                  stroke="currentColor"
                  viewBox="0 0 24 24"
                >
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    strokeWidth="2"
                    d="M6 18L18 6M6 6l12 12"
                  />
                </svg>
              </button>
            </div>

            {/* Timer and Progress - More Compact */}
            <div
              className={`bg-gradient-to-r ${getTimerBgGradient()} rounded-xl p-2. 5 border ${
                secondsRemaining !== null && secondsRemaining < 60
                  ? "border-red-400 animate-pulse"
                  : "border-white/30"
              }`}
            >
              <div className="flex items-center justify-between gap-3">
                <div className="flex items-center gap-2">
                  <div className="relative">
                    <div className="w-9 h-9 bg-white rounded-lg shadow-lg flex items-center justify-center">
                      <svg
                        className={`w-5 h-5 ${getTimerColor()}`}
                        fill="none"
                        stroke="currentColor"
                        viewBox="0 0 24 24"
                      >
                        <path
                          strokeLinecap="round"
                          strokeLinejoin="round"
                          strokeWidth="2"
                          d="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z"
                        />
                      </svg>
                    </div>
                    {secondsRemaining !== null && secondsRemaining < 60 && (
                      <span className="absolute -top-1 -right-1 flex h-2. 5 w-2.5">
                        <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-red-400 opacity-75"></span>
                        <span className="relative inline-flex rounded-full h-2.5 w-2.5 bg-red-500"></span>
                      </span>
                    )}
                  </div>
                  <div>
                    <p className="text-xs text-gray-700 font-semibold leading-tight">
                      Time Left
                    </p>
                    <p
                      className={`font-mono text-base sm:text-lg font-bold ${getTimerColor()} leading-tight`}
                    >
                      {formatTime(secondsRemaining || 0)}
                    </p>
                  </div>
                </div>

                <div className="text-right">
                  <p className="text-xs text-gray-700 font-semibold leading-tight">
                    Answered
                  </p>
                  <div className="flex items-baseline gap-1">
                    <span className="text-base sm:text-lg font-bold text-gray-900 leading-tight">
                      {answeredQuestions}
                    </span>
                    <span className="text-xs text-gray-600">
                      / {questions.length}
                    </span>
                  </div>
                </div>
              </div>

              {/* Progress Bar - Inline */}
              <div className="mt-1. 5">
                <div className="h-1 bg-white/20 rounded-full overflow-hidden backdrop-blur-sm">
                  <div
                    className="h-full bg-white rounded-full transition-all duration-500 shadow-lg"
                    style={{ width: `${progressPercentage}%` }}
                  ></div>
                </div>
              </div>
            </div>
          </div>
        </div>

        {/* Scrollable Question Area - Maximum Height with better spacing */}
        <div
          className="flex-1 overflow-y-auto custom-scrollbar"
          style={{ minHeight: 0 }}
        >
          <div className="p-4 sm:p-5">
            {/* Question - More Compact */}
            <div className="mb-4">
              <div className="flex items-start gap-2. 5 mb-4">
                <div className="w-9 h-9 bg-gradient-to-br from-green-500 to-emerald-600 text-white rounded-lg flex items-center justify-center font-bold text-base flex-shrink-0 shadow-lg">
                  {currentQuestion + 1}
                </div>
                <h3 className="text-base sm:text-lg font-bold text-gray-900 flex-1 leading-snug pt-0.5">
                  {currentQ.question}
                </h3>
              </div>

              {/* Options - Smaller */}
              <div className="space-y-2.5">
                {currentQ.options.map((option, index) => {
                  const isSelected = selectedAnswers[currentQuestion] === index;
                  const optionLetters = ["A", "B", "C", "D", "E", "F"];

                  return (
                    <button
                      key={index}
                      onClick={() => handleAnswerSelect(index)}
                      className={`w-full p-3 rounded-xl border-2 text-left transition-all duration-300 group ${
                        isSelected
                          ? "border-green-500 bg-gradient-to-r from-green-50 to-emerald-50 shadow-md transform scale-[1.01]"
                          : "border-gray-200 hover:border-green-300 hover:bg-gray-50 hover:shadow-sm"
                      }`}
                    >
                      <div className="flex items-center gap-2. 5">
                        <div
                          className={`w-8 h-8 rounded-lg border-2 flex items-center justify-center font-bold text-sm transition-all duration-300 flex-shrink-0 ${
                            isSelected
                              ? "border-green-500 bg-gradient-to-br from-green-500 to-emerald-600 text-white shadow-lg scale-110"
                              : "border-gray-300 text-gray-500 bg-white group-hover:border-green-400 group-hover:scale-105"
                          }`}
                        >
                          {isSelected ? (
                            <svg
                              className="w-5 h-5"
                              fill="none"
                              stroke="currentColor"
                              viewBox="0 0 24 24"
                            >
                              <path
                                strokeLinecap="round"
                                strokeLinejoin="round"
                                strokeWidth="3"
                                d="M5 13l4 4L19 7"
                              />
                            </svg>
                          ) : (
                            optionLetters[index]
                          )}
                        </div>
                        <span className="font-medium text-gray-900 flex-1 text-sm leading-snug">
                          {option}
                        </span>
                      </div>
                    </button>
                  );
                })}
              </div>
            </div>

            {/* Low Time Warning - Smaller */}
            {secondsRemaining !== null && secondsRemaining < 60 && (
              <div className="bg-gradient-to-r from-red-50 to-orange-50 border-2 border-red-300 rounded-xl p-3 flex items-start gap-2.5 animate-pulse shadow-lg">
                <div className="w-8 h-8 bg-gradient-to-br from-red-500 to-orange-600 rounded-lg flex items-center justify-center flex-shrink-0">
                  <svg
                    className="w-4 h-4 text-white"
                    fill="none"
                    stroke="currentColor"
                    viewBox="0 0 24 24"
                  >
                    <path
                      strokeLinecap="round"
                      strokeLinejoin="round"
                      strokeWidth="2"
                      d="M12 9v2m0 4h. 01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-. 77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z"
                    />
                  </svg>
                </div>
                <div className="flex-1">
                  <p className="text-red-800 font-bold text-sm">
                    ⚠️ Time Running Out!
                  </p>
                  <p className="text-red-700 text-xs leading-snug">
                    Less than 1 minute remaining. Submit soon!{" "}
                  </p>
                </div>
              </div>
            )}
          </div>
        </div>

        {/* Fixed Footer Navigation - More Compact */}
        <div className="border-t bg-gradient-to-r from-gray-50 to-gray-100 px-4 sm:px-6 py-2.5 flex justify-between items-center flex-shrink-0 shadow-lg">
          <Button
            onClick={handlePrevious}
            disabled={currentQuestion === 0}
            variant="outline"
            className="disabled:opacity-50 disabled:cursor-not-allowed px-3 sm:px-4 py-2 border-2 font-semibold text-sm"
          >
            <svg
              className="w-4 h-4 sm:mr-1. 5"
              fill="none"
              stroke="currentColor"
              viewBox="0 0 24 24"
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth="2"
                d="M15 19l-7-7 7-7"
              />
            </svg>
            <span className="hidden sm:inline">Previous</span>
          </Button>

          <div className="text-center px-2">
            <div className="flex gap-1">
              {questions.map((_, idx) => (
                <div
                  key={idx}
                  className={`w-1. 5 h-1.5 rounded-full transition-all ${
                    selectedAnswers[idx] !== undefined
                      ? "bg-green-500 w-2.5"
                      : idx === currentQuestion
                      ? "bg-blue-500 w-2.5"
                      : "bg-gray-300"
                  }`}
                  title={`Question ${idx + 1}`}
                />
              ))}
            </div>
          </div>

          <Button
            onClick={handleNext}
            disabled={selectedAnswers[currentQuestion] === undefined}
            className="bg-gradient-to-r from-green-500 to-emerald-600 text-white disabled:opacity-50 disabled:cursor-not-allowed px-3 sm:px-4 py-2 font-semibold shadow-lg hover:shadow-xl transform hover:scale-105 transition-all text-sm"
          >
            <span className="hidden sm:inline">
              {currentQuestion === questions.length - 1 ? "Finish" : "Next"}
            </span>
            <span className="sm:hidden">
              {currentQuestion === questions.length - 1 ? "✓" : "→"}
            </span>
            <svg
              className="w-4 h-4 ml-1 sm:ml-1.5 hidden sm:inline"
              fill="none"
              stroke="currentColor"
              viewBox="0 0 24 24"
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth="2"
                d={
                  currentQuestion === questions.length - 1
                    ? "M5 13l4 4L19 7"
                    : "M9 5l7 7-7 7"
                }
              />
            </svg>
          </Button>
        </div>
      </div>

      <style jsx>{`
        .custom-scrollbar::-webkit-scrollbar {
          width: 6px;
        }
        .custom-scrollbar::-webkit-scrollbar-track {
          background: #f1f5f9;
          border-radius: 10px;
        }
        . custom-scrollbar::-webkit-scrollbar-thumb {
          background: linear-gradient(to bottom, #10b981, #059669);
          border-radius: 10px;
        }
        .custom-scrollbar::-webkit-scrollbar-thumb:hover {
          background: linear-gradient(to bottom, #059669, #047857);
        }
        @keyframes fadeIn {
          from {
            opacity: 0;
          }
          to {
            opacity: 1;
          }
        }
        @keyframes slideUp {
          from {
            transform: translateY(20px);
            opacity: 0;
          }
          to {
            transform: translateY(0);
            opacity: 1;
          }
        }
        .animate-fadeIn {
          animation: fadeIn 0.3s ease-out;
        }
        .animate-slideUp {
          animation: slideUp 0.4s ease-out;
        }
      `}</style>
    </div>
  );
}

export default QuizModal;
