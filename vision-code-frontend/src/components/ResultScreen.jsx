import {
  ArrowLeft,
  CheckCircle,
  XCircle,
  Trophy,
  AlertCircle,
  RotateCcw,
  BarChart3,
  ShieldCheck,
  ShieldAlert,
} from "lucide-react";

export default function ResultScreen({
  questions,
  answers,
  quizData,
  title,
  timeTaken,
  totalTime,
  onRetry,
  onBack,
  report,
  calcScore,
  formatTime,
}) {
  const score = calcScore(questions, answers);
  const total = questions.length;
  const pct = Math.round((score / total) * 100);

  // -------------------------------
  // ✅ SAFE DATA EXTRACTION
  // -------------------------------
  const scorePass = score >= quizData.passing_marks;

  const integrityScore = report?.integrity_verdict?.score || 0;
  const integrityPass = integrityScore >= 70;

  const faceFails = report?.behavioral_events?.face_verifications_fail || 0;
  const alerts = report?.behavioral_events?.total_alerts || 0;

  // -------------------------------
  // ✅ RESULT LOGIC (FIXED)
  // -------------------------------
  let result = "FAIL";

  if (scorePass && integrityScore >= 80 && faceFails === 0 && alerts <= 2) {
    result = "EXCELLENT";
  } else if (scorePass && integrityPass) {
    result = "PASS";
  } else if (scorePass && !integrityPass) {
    result = "SUSPICIOUS";
  }

  // -------------------------------
  // 🎨 UI HELPERS
  // -------------------------------
  const resultColor = {
    EXCELLENT: "text-green-700",
    PASS: "text-blue-700",
    SUSPICIOUS: "text-yellow-600",
    FAIL: "text-red-700",
  };

  const bgColor = {
    EXCELLENT: "bg-green-100",
    PASS: "bg-blue-100",
    SUSPICIOUS: "bg-yellow-100",
    FAIL: "bg-red-100",
  };

  const barColor = {
    EXCELLENT: "bg-green-500",
    PASS: "bg-blue-500",
    SUSPICIOUS: "bg-yellow-500",
    FAIL: "bg-red-500",
  };

  return (
    <div className="min-h-screen bg-gray-50">
      <div className="max-w-3xl mx-auto px-4 py-8">
        {/* Back Button */}
        <button
          onClick={onBack}
          className="inline-flex items-center gap-2 text-gray-600 hover:text-blue-600 transition mb-8"
        >
          <ArrowLeft className="w-4 h-4" /> Back to Course
        </button>

        <div className="bg-white rounded-2xl shadow-sm border border-gray-100 p-8 mb-8">
          {/* ---------------- HEADER ---------------- */}
          <div className="text-center mb-8">
            <div
              className={`w-20 h-20 mx-auto rounded-full flex items-center justify-center mb-6 ${bgColor[result]}`}
            >
              {(result === "EXCELLENT" || result === "PASS") && (
                <Trophy className="w-10 h-10 text-green-600" />
              )}
              {result === "SUSPICIOUS" && (
                <ShieldAlert className="w-10 h-10 text-yellow-600" />
              )}
              {result === "FAIL" && (
                <XCircle className="w-10 h-10 text-red-600" />
              )}
            </div>

            <h1 className={`text-3xl font-bold mb-2 ${resultColor[result]}`}>
              {result === "EXCELLENT" && "Excellent Performance! 🏆"}
              {result === "PASS" && "Quiz Passed! 🎉"}
              {result === "SUSPICIOUS" && "Passed (Review Required) ⚠️"}
              {result === "FAIL" && "Quiz Failed ❌"}
            </h1>

            <p className="text-gray-600 text-sm">{title}</p>

            {result === "SUSPICIOUS" && (
              <p className="text-xs text-yellow-600 mt-2">
                Unusual behavior detected during exam
              </p>
            )}
          </div>

          {/* ---------------- SCORE ---------------- */}
          <div className="bg-gradient-to-r from-blue-50 to-purple-50 rounded-xl p-8 mb-8">
            <div className="text-center mb-6">
              <div className="flex items-center justify-center gap-2 mb-4">
                <span className="text-6xl font-bold text-gray-900">
                  {score}
                </span>
                <span className="text-2xl text-gray-500">/ {total}</span>
              </div>
              <div className="text-3xl font-bold text-blue-600">{pct}%</div>
            </div>

            <div className="w-full h-3 bg-gray-200 rounded-full overflow-hidden mb-6">
              <div
                className={`h-full rounded-full transition-all duration-700 ${barColor[result]}`}
                style={{ width: `${pct}%` }}
              />
            </div>

            <div className="grid grid-cols-3 gap-4 text-sm">
              {[
                [
                  "Passing Score",
                  `${Math.round((quizData.passing_marks / total) * 100)}%`,
                ],
                ["Your Score", `${pct}%`],
                ["Time Taken", formatTime(timeTaken)],
              ].map(([label, val]) => (
                <div
                  key={label}
                  className="bg-white rounded-lg p-3 text-center"
                >
                  <p className="text-gray-500 text-xs mb-1">{label}</p>
                  <p className="font-bold text-gray-900">{val}</p>
                </div>
              ))}
            </div>
          </div>

          {/* ---------------- PROCTORING SUMMARY ---------------- */}
          {report?.integrity_verdict && (
            <div className="mb-8 border border-gray-200 rounded-xl overflow-hidden">
              <div className="bg-gray-50 px-4 py-3 border-b flex items-center gap-2">
                <ShieldCheck className="w-4 h-4 text-blue-600" />
                <span className="text-sm font-semibold">
                  Proctoring Summary
                </span>
              </div>

              <div className="p-6 grid grid-cols-2 md:grid-cols-3 gap-4 text-sm">
                <Card label="Integrity Score" value={integrityScore} />
                <Card
                  label="Attention"
                  value={
                    report?.metrics?.attentive_pct
                      ? report.metrics.attentive_pct.toFixed(1) + "%"
                      : "0%"
                  }
                />
                <Card
                  label="AI Accuracy"
                  value={
                    report?.metrics?.accuracy
                      ? (report.metrics.accuracy * 100).toFixed(1) + "%"
                      : "0%"
                  }
                />
                <Card label="Alerts" value={alerts} />
                <Card
                  label="Noise"
                  value={report?.behavioral_events?.sound_detections || 0}
                />
                <Card label="Face Fails" value={faceFails} />
              </div>

              {/* Flags */}
              {report?.integrity_verdict?.flags?.length > 0 && (
                <div className="px-6 pb-4">
                  <div className="bg-yellow-50 border border-yellow-200 rounded-lg p-3 text-xs text-yellow-800">
                    ⚠ {report.integrity_verdict.flags.join(", ")}
                  </div>
                </div>
              )}
            </div>
          )}

          {/* ---------------- ANSWER REVIEW ---------------- */}
          <div className="mb-8">
            <h2 className="text-xl font-bold mb-4 flex items-center gap-2">
              <BarChart3 className="w-5 h-5 text-blue-600" /> Answer Review
            </h2>

            <div className="space-y-3">
              {questions.map((q, idx) => {
                const chosen = q.options.find((o) => o.id === answers[q.id]);
                const isRight = chosen?.is_correct ?? false;

                return (
                  <div
                    key={q.id}
                    className={`border rounded-lg p-4 ${
                      isRight
                        ? "bg-green-50 border-green-200"
                        : "bg-red-50 border-red-200"
                    }`}
                  >
                    <div className="flex gap-3 mb-3">
                      {isRight ? (
                        <CheckCircle className="w-5 h-5 text-green-600" />
                      ) : (
                        <XCircle className="w-5 h-5 text-red-600" />
                      )}
                      <p className="font-semibold text-sm">
                        Q{idx + 1}. {q.question_text}
                      </p>
                    </div>

                    <div className="ml-8 space-y-2">
                      {q.options.map((opt) => {
                        const isChosen = opt.id === answers[q.id];
                        const isCorrect = opt.is_correct;

                        let cls = "bg-white border-gray-200 text-gray-700";
                        if (isCorrect)
                          cls = "bg-green-100 border-green-300 text-green-800";
                        else if (isChosen)
                          cls = "bg-red-100 border-red-300 text-red-800";

                        return (
                          <div
                            key={opt.id}
                            className={`text-xs px-3 py-2 rounded border ${cls}`}
                          >
                            {opt.option_text}
                          </div>
                        );
                      })}
                    </div>
                  </div>
                );
              })}
            </div>
          </div>

          {/* ---------------- BUTTONS ---------------- */}
          <div className="flex gap-3">
            <button
              onClick={onBack}
              className="flex-1 px-4 py-3 border rounded-xl font-semibold hover:bg-gray-50"
            >
              Back to Course
            </button>
            <button
              onClick={onRetry}
              className="flex-1 px-4 py-3 bg-blue-600 text-white rounded-xl font-semibold hover:bg-blue-700"
            >
              Retry Quiz
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}

// -------------------------------
// ✅ REUSABLE CARD COMPONENT
// -------------------------------
function Card({ label, value }) {
  return (
    <div className="bg-white border rounded-lg p-4 text-center">
      <p className="text-gray-500 text-xs">{label}</p>
      <p className="text-xl font-bold">{value}</p>
    </div>
  );
}
