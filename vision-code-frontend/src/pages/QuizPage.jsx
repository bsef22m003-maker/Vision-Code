import { useState, useEffect, useCallback, useRef } from "react";
import { useLocation, useNavigate, useParams } from "react-router-dom";
import {
  ArrowLeft,
  Clock,
  CheckCircle,
  XCircle,
  Trophy,
  AlertCircle,
  ChevronRight,
  ChevronLeft,
  RotateCcw,
  BarChart3,
  Eye,
  EyeOff,
  ShieldCheck,
  ShieldAlert,
  Loader2,
  Camera,
  Crosshair,
  Play,
} from "lucide-react";
import ResultScreen from "../components/ResultScreen";

// ─── Config ───────────────────────────────────────────────────────────────────
const BASE_URL = "https://examproctoringapi-production.up.railway.app";
const POLL_INTERVAL_MS = 2000;
const FREEZE_EVENTS = ["INATTENTIVE", "EYES_OFF_SCREEN"];

// ─── Helpers ──────────────────────────────────────────────────────────────────
function formatTime(seconds) {
  const m = Math.floor(seconds / 60);
  const s = seconds % 60;
  return `${String(m).padStart(2, "0")}:${String(s).padStart(2, "0")}`;
}

function calcScore(questions, answers) {
  return questions.reduce((acc, q) => {
    const chosen = q.options.find((o) => o.id === answers[q.id]);
    return acc + (chosen?.is_correct ? 1 : 0);
  }, 0);
}

function isStateInattentive(currentState) {
  if (!currentState) return false;
  const attention = (currentState.attention ?? "").toUpperCase();
  const status = (currentState.candidate_status ?? "").toUpperCase();
  return (
    attention === "INATTENTIVE" ||
    status.includes("INATTENTIVE") ||
    status.includes("EYES") ||
    status.includes("AWAY") ||
    status.includes("DEVIAT")
  );
}

function parseSessionStatus(data) {
  if (!data || typeof data !== "object")
    return { faceRegistered: false, calibrated: false };
  const status = data.status ?? "";
  return {
    faceRegistered: ["calibrating", "active", "ended"].includes(status),
    calibrated: ["active", "ended"].includes(status),
  };
}

// ─── Timer Bar ────────────────────────────────────────────────────────────────
function TimerBar({ timeLeft, totalTime }) {
  const pct = Math.max(0, (timeLeft / totalTime) * 100);
  const urgent = pct < 25;
  const warning = pct < 50;
  return (
    <div className="w-full h-2 bg-gray-100 rounded-full overflow-hidden">
      <div
        className={`h-full rounded-full transition-all duration-1000 ${
          urgent ? "bg-red-500" : warning ? "bg-yellow-500" : "bg-blue-600"
        }`}
        style={{ width: `${pct}%` }}
      />
    </div>
  );
}

// ─── Setup Step ───────────────────────────────────────────────────────────────
function SetupStep({ num, label, sublabel, status }) {
  return (
    <div style={{ display: "flex", alignItems: "center", gap: 12 }}>
      <div
        style={{
          width: 36,
          height: 36,
          borderRadius: "50%",
          flexShrink: 0,
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
          fontWeight: 700,
          fontSize: 14,
          background:
            status === "done"
              ? "#dcfce7"
              : status === "active"
                ? "#dbeafe"
                : "#f3f4f6",
          border: `2px solid ${status === "done" ? "#22c55e" : status === "active" ? "#3b82f6" : "#e5e7eb"}`,
          color:
            status === "done"
              ? "#16a34a"
              : status === "active"
                ? "#2563eb"
                : "#9ca3af",
          transition: "all 0.4s",
        }}
      >
        {status === "done" ? (
          <CheckCircle style={{ width: 18, height: 18 }} />
        ) : (
          num
        )}
      </div>
      <div style={{ flex: 1 }}>
        <div
          style={{
            fontSize: 13,
            fontWeight: 600,
            color: status === "waiting" ? "#9ca3af" : "#111827",
          }}
        >
          {label}
        </div>
        <div style={{ fontSize: 11, color: "#9ca3af" }}>{sublabel}</div>
      </div>
      {status === "active" && (
        <Loader2
          style={{
            width: 16,
            height: 16,
            color: "#3b82f6",
            animation: "spin 1s linear infinite",
            flexShrink: 0,
          }}
        />
      )}
      {status === "done" && (
        <span
          style={{
            fontSize: 11,
            color: "#16a34a",
            fontWeight: 600,
            flexShrink: 0,
          }}
        >
          Done
        </span>
      )}
    </div>
  );
}

// ─── Result Screen ────────────────────────────────────────────────────────────

// ─── Main Quiz Page ───────────────────────────────────────────────────────────
export default function QuizPage() {
  const { quizItemId } = useParams();
  const location = useLocation();
  const navigate = useNavigate();

  const locationState = location.state;
  const quizData = locationState?.quizData ?? null;
  const title = locationState?.title ?? "Quiz";
  const courseId = locationState?.courseId;
  const totalTime = (locationState?.estimatedMinutes ?? 10) * 60;

  const questions = quizData?.questions ?? [];

  // ── Quiz state ─────────────────────────────────────────────────────────────
  const [currentIdx, setCurrentIdx] = useState(0);
  const [answers, setAnswers] = useState({});
  const [submitted, setSubmitted] = useState(false);
  const [timeLeft, setTimeLeft] = useState(totalTime);
  const [timeTaken, setTimeTaken] = useState(0);
  const [quizStarted, setQuizStarted] = useState(false);
  const timerRef = useRef(null);

  // ── Proctoring state ───────────────────────────────────────────────────────
  const [sessionId, setSessionId] = useState(null);
  const [sessionReady, setSessionReady] = useState(false);
  const [alerts, setAlerts] = useState([]);
  const [report, setReport] = useState(null);
  const [attentionState, setAttentionState] = useState("UNKNOWN");
  const [isFrozen, setIsFrozen] = useState(false);
  const hasEndedRef = useRef(false);

  // ── Setup state ────────────────────────────────────────────────────────────
  const [setupStep, setSetupStep] = useState(0);
  const [calibrationReady, setCalibrationReady] = useState(false);
  const [showInfo, setShowInfo] = useState(true);

  // Refs so closures always read fresh values without re-subscribing
  const setupStepRef = useRef(0);
  const quizStartedRef = useRef(false);
  useEffect(() => {
    setupStepRef.current = setupStep;
  }, [setupStep]);
  useEffect(() => {
    quizStartedRef.current = quizStarted;
  }, [quizStarted]);

  // ── Create session when leaving info screen ────────────────────────────────
  useEffect(() => {
    if (showInfo) return;
    const createSession = async () => {
      try {
        const res = await fetch(`${BASE_URL}/api/sessions`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            candidate_id: "cand_001",
            candidate_name: "Test User",
            exam_id: "exam_001",
            exam_label: title,
          }),
        });
        const data = await res.json();
        setSessionId(data.session_id);
        setSessionReady(true);
      } catch (err) {
        console.error("Session creation failed:", err);
      }
    };
    createSession();
  }, [showInfo, title]);

  // ── Poll session status for setup step detection ───────────────────────────
  useEffect(() => {
    if (!sessionId) return;
    const poll = async () => {
      try {
        const res = await fetch(`${BASE_URL}/api/sessions/${sessionId}`);
        if (!res.ok) return;
        const data = await res.json();
        const { faceRegistered, calibrated } = parseSessionStatus(data);
        if (faceRegistered && setupStepRef.current === 0) setSetupStep(1);
        if (calibrated && setupStepRef.current >= 1) {
          setSetupStep(2);
          setCalibrationReady(true);
        }
        const attention = data.current_state?.attention;
        if (attention) setAttentionState(attention);
      } catch {
        /* silent */
      }
    };
    poll();
    const id = setInterval(poll, POLL_INTERVAL_MS);
    return () => clearInterval(id);
  }, [sessionId]);

  // ── postMessage — primary attention signal from the iframe ─────────────────
  useEffect(() => {
    const allowedOrigin = (() => {
      try {
        return new URL(BASE_URL).origin;
      } catch {
        return BASE_URL;
      }
    })();

    const handler = (event) => {
      if (!event.data || typeof event.data !== "object") return;
      if (event.origin !== allowedOrigin) return;

      const data = event.data;
      const type = (data.type ?? "").toLowerCase();
      const step = setupStepRef.current;

      // Setup step progression
      const isFaceEvent =
        type.includes("face") ||
        type.includes("registr") ||
        type === "step_1_complete" ||
        type === "step1_complete";
      if (isFaceEvent && step === 0) setSetupStep(1);

      const isCalibEvent =
        type.includes("calibrat") ||
        type.includes("setup_complete") ||
        type === "ready" ||
        type === "gaze_calibrated" ||
        type === "step_2_complete" ||
        type === "step2_complete" ||
        type === "complete";
      if (isCalibEvent && step >= 1) {
        setSetupStep(2);
        setCalibrationReady(true);
      }

      if (
        step === 0 &&
        !isFaceEvent &&
        (type.includes("complet") ||
          type.includes("done") ||
          type.includes("success") ||
          type.includes("ready"))
      ) {
        setSetupStep(1);
      }

      // Metrics — drives all attention state and freeze logic
      if (type === "metrics") {
        const cs = data.state ?? data.current_state ?? {};
        const inattentive = isStateInattentive(cs);
        const attention = (cs.attention ?? "").toUpperCase();

        if (attention && attention !== "UNKNOWN" && step >= 1) {
          setSetupStep(2);
          setCalibrationReady(true);
        }

        setAttentionState(inattentive ? "INATTENTIVE" : attention || "UNKNOWN");

        if (quizStartedRef.current) {
          if (inattentive) setIsFrozen(true);
          else if (attention === "ATTENTIVE") setIsFrozen(false);
        }
      }

      if (type === "alert" || type === "integrity_alert") {
        setAlerts((prev) => [...prev, data]);
        if (quizStartedRef.current && FREEZE_EVENTS.includes(data.event)) {
          setIsFrozen(true);
          setAttentionState("INATTENTIVE");
        }
      }

      if (type === "session_ended" || type === "session_complete") {
        setReport(data.report ?? null);
      }
    };

    window.addEventListener("message", handler);
    return () => window.removeEventListener("message", handler);
  }, []); // empty — refs keep values fresh

  // ── Timer — pauses when frozen ─────────────────────────────────────────────
  const stopTimer = useCallback(() => {
    if (timerRef.current) clearInterval(timerRef.current);
  }, []);

  useEffect(() => {
    if (!quizStarted || submitted || isFrozen) {
      stopTimer();
      return;
    }
    timerRef.current = setInterval(() => {
      setTimeLeft((t) => {
        if (t <= 1) {
          stopTimer();
          setTimeTaken(totalTime);
          setSubmitted(true);
          return 0;
        }
        return t - 1;
      });
    }, 1000);
    return stopTimer;
  }, [quizStarted, submitted, isFrozen, stopTimer, totalTime]);

  // ── End session ────────────────────────────────────────────────────────────
  const endProctoringSession = useCallback(async () => {
    if (!sessionId || hasEndedRef.current) return;
    hasEndedRef.current = true;
    try {
      await fetch(`${BASE_URL}/api/sessions/${sessionId}/end`, {
        method: "POST",
      });
      const res = await fetch(`${BASE_URL}/api/sessions/${sessionId}/report`);
      setReport(await res.json());
    } catch (err) {
      console.error("End session error:", err);
    }
  }, [sessionId]);

  const handleSubmit = () => {
    stopTimer();
    setTimeTaken(totalTime - timeLeft);
    setSubmitted(true);
    endProctoringSession();
  };

  const handleRetry = () => {
    hasEndedRef.current = false;
    setAnswers({});
    setSubmitted(false);
    setCurrentIdx(0);
    setTimeLeft(totalTime);
    setTimeTaken(0);
    setQuizStarted(false);
    setIsFrozen(false);
    setAlerts([]);
    setReport(null);
    setAttentionState("UNKNOWN");
    setSetupStep(0);
    setCalibrationReady(false);
    setSessionId(null);
    setSessionReady(false);
    setShowInfo(false);
  };

  const handleBack = () =>
    navigate(courseId ? `/student/course/${courseId}` : "/student");

  // ── Guards ─────────────────────────────────────────────────────────────────
  if (!quizData) {
    return (
      <div className="min-h-screen bg-gray-50 flex items-center justify-center">
        <div className="bg-white rounded-2xl shadow-sm border border-gray-100 p-8 text-center max-w-md">
          <AlertCircle className="w-12 h-12 text-gray-400 mx-auto mb-4" />
          <p className="text-lg font-bold text-gray-900 mb-4">
            Quiz data not found.
          </p>
          <button
            onClick={() => navigate(-1)}
            className="w-full px-4 py-3 bg-blue-600 text-white rounded-xl font-semibold hover:bg-blue-700 transition"
          >
            Go Back
          </button>
        </div>
      </div>
    );
  }

  if (submitted) {
    return (
      <ResultScreen
        questions={questions}
        answers={answers}
        quizData={quizData}
        title={title}
        timeTaken={timeTaken}
        totalTime={totalTime}
        onRetry={handleRetry}
        onBack={handleBack}
        report={report}
        calcScore={calcScore}
        formatTime={formatTime}
      />
    );
  }

  // ── Info screen ────────────────────────────────────────────────────────────
  if (showInfo) {
    return (
      <div className="min-h-screen bg-gray-50 flex items-center justify-center px-4">
        <div className="bg-white rounded-2xl shadow-sm border border-gray-100 p-8 max-w-md w-full">
          <div className="w-16 h-16 mx-auto bg-blue-100 rounded-full flex items-center justify-center mb-6">
            <Clock className="w-8 h-8 text-blue-600" />
          </div>
          <h1 className="text-2xl font-bold text-gray-900 text-center mb-2">
            {title}
          </h1>
          <p className="text-gray-600 text-sm text-center mb-6">
            Ready to test your knowledge?
          </p>
          <div className="bg-blue-50 border border-blue-100 rounded-xl p-4 mb-6 flex items-start gap-3">
            <ShieldCheck className="w-5 h-5 text-blue-600 flex-shrink-0 mt-0.5" />
            <div>
              <p className="text-sm font-semibold text-blue-800 mb-1">
                AI Proctoring Enabled
              </p>
              <p className="text-xs text-blue-700 leading-relaxed">
                You'll set up your camera before the exam begins. Looking away
                will <strong>pause</strong> the exam and timer until you face
                the screen again.
              </p>
            </div>
          </div>
          <div className="space-y-3 mb-8">
            {[
              ["Questions", questions.length],
              ["Time Limit", formatTime(totalTime)],
              [
                "Passing Score",
                `${quizData.passing_marks}/${quizData.total_marks}`,
              ],
            ].map(([label, val]) => (
              <div
                key={label}
                className="bg-gray-50 rounded-lg p-4 flex items-center justify-between"
              >
                <span className="text-sm text-gray-600 font-medium">
                  {label}
                </span>
                <span className="text-xl font-bold text-blue-600">{val}</span>
              </div>
            ))}
          </div>
          <button
            onClick={() => setShowInfo(false)}
            className="w-full px-4 py-3 bg-blue-600 text-white rounded-xl font-semibold hover:bg-blue-700 transition mb-3 flex items-center justify-center gap-2"
          >
            <Camera className="w-4 h-4" /> Set Up Camera & Start
          </button>
          <button
            onClick={handleBack}
            className="w-full px-4 py-3 border border-gray-200 text-gray-700 rounded-xl font-semibold hover:bg-gray-50 transition"
          >
            Cancel
          </button>
        </div>
      </div>
    );
  }

  // ── Setup + Quiz — SINGLE layout, iframe never moves ──────────────────────
  const embedUrl = sessionId ? `${BASE_URL}/?session_id=${sessionId}` : null;
  const step0Status =
    setupStep === 0 ? "active" : setupStep > 0 ? "done" : "waiting";
  const step1Status =
    setupStep === 1 ? "active" : setupStep > 1 ? "done" : "waiting";
  const currentQ = questions[currentIdx];
  const answeredCount = Object.keys(answers).length;
  const isLast = currentIdx === questions.length - 1;
  const urgent = timeLeft < totalTime * 0.25;
  const attentionColor =
    attentionState === "ATTENTIVE"
      ? "#16a34a"
      : attentionState === "INATTENTIVE"
        ? "#dc2626"
        : "#9ca3af";
  const attentionLabel =
    attentionState === "ATTENTIVE"
      ? "Attentive"
      : attentionState === "INATTENTIVE"
        ? "Inattentive"
        : "Monitoring…";

  return (
    <div style={{ minHeight: "100vh", background: "#f9fafb" }}>
      <div
        style={{
          maxWidth: 1100,
          margin: "0 auto",
          padding: "32px 20px",
          display: "flex",
          gap: 24,
          alignItems: "flex-start",
        }}
      >
        {/* ── LEFT: toggles between setup UI and quiz questions ─────────── */}
        <div style={{ flex: 1, minWidth: 0 }}>
          {/* SETUP UI */}
          {!quizStarted && (
            <div style={{ display: "flex", flexDirection: "column", gap: 14 }}>
              <button
                onClick={handleBack}
                className="inline-flex items-center gap-2 text-gray-600 hover:text-blue-600 transition self-start"
              >
                <ArrowLeft className="w-4 h-4" /> Back to Course
              </button>

              <div
                style={{
                  background: "#fff",
                  borderRadius: 14,
                  border: "1px solid #e5e7eb",
                  padding: "18px 20px",
                }}
              >
                <div
                  style={{
                    display: "flex",
                    alignItems: "center",
                    gap: 8,
                    marginBottom: 6,
                  }}
                >
                  <ShieldCheck
                    style={{ width: 18, height: 18, color: "#2563eb" }}
                  />
                  <span
                    style={{ fontWeight: 700, fontSize: 14, color: "#111827" }}
                  >
                    Camera Setup Required
                  </span>
                </div>
                <p
                  style={{
                    fontSize: 12,
                    color: "#6b7280",
                    margin: 0,
                    lineHeight: 1.6,
                  }}
                >
                  Complete both steps in the camera panel on the right. The quiz
                  will unlock automatically once done.
                </p>
              </div>

              <div
                style={{
                  background: "#fff",
                  borderRadius: 14,
                  border: "1px solid #e5e7eb",
                  padding: "18px 20px",
                  display: "flex",
                  flexDirection: "column",
                  gap: 16,
                }}
              >
                <SetupStep
                  num={1}
                  label="Face Registration"
                  sublabel='Center face → click "Capture Face"'
                  status={step0Status}
                />
                <div style={{ height: 1, background: "#f3f4f6" }} />
                <SetupStep
                  num={2}
                  label="Gaze Calibration"
                  sublabel='Click "Begin" → follow the dot'
                  status={step1Status}
                />
              </div>

              <div
                style={{
                  background:
                    setupStep === 0
                      ? "#eff6ff"
                      : setupStep === 1
                        ? "#fefce8"
                        : "#f0fdf4",
                  border: `1px solid ${setupStep === 0 ? "#bfdbfe" : setupStep === 1 ? "#fef08a" : "#bbf7d0"}`,
                  borderRadius: 12,
                  padding: "14px 16px",
                  display: "flex",
                  gap: 10,
                  alignItems: "flex-start",
                  transition: "all 0.4s",
                }}
              >
                {setupStep === 0 && (
                  <Camera
                    style={{
                      width: 18,
                      height: 18,
                      color: "#2563eb",
                      flexShrink: 0,
                      marginTop: 1,
                    }}
                  />
                )}
                {setupStep === 1 && (
                  <Crosshair
                    style={{
                      width: 18,
                      height: 18,
                      color: "#d97706",
                      flexShrink: 0,
                      marginTop: 1,
                    }}
                  />
                )}
                {setupStep === 2 && (
                  <CheckCircle
                    style={{
                      width: 18,
                      height: 18,
                      color: "#16a34a",
                      flexShrink: 0,
                      marginTop: 1,
                    }}
                  />
                )}
                <div>
                  <p
                    style={{
                      fontSize: 13,
                      fontWeight: 600,
                      margin: "0 0 3px",
                      color:
                        setupStep === 0
                          ? "#1d4ed8"
                          : setupStep === 1
                            ? "#b45309"
                            : "#15803d",
                    }}
                  >
                    {setupStep === 0 && "Step 1: Register your face"}
                    {setupStep === 1 && "Step 2: Calibrate your gaze"}
                    {setupStep === 2 && "All done! Ready to start."}
                  </p>
                  <p
                    style={{
                      fontSize: 12,
                      color: "#6b7280",
                      margin: 0,
                      lineHeight: 1.5,
                    }}
                  >
                    {setupStep === 0 &&
                      "Look at the camera, center your face in the oval shown in the right panel, then click Capture Face."}
                    {setupStep === 1 &&
                      "Click Begin Calibration in the right panel, then follow the moving dot with your eyes until it disappears."}
                    {setupStep === 2 &&
                      "Face registered and gaze calibrated. Click Begin Exam when you're ready."}
                  </p>
                </div>
              </div>

              <button
                onClick={() => setQuizStarted(true)}
                disabled={!calibrationReady}
                style={{
                  width: "100%",
                  padding: "13px 0",
                  borderRadius: 12,
                  border: "none",
                  fontWeight: 700,
                  fontSize: 14,
                  cursor: calibrationReady ? "pointer" : "not-allowed",
                  background: calibrationReady ? "#2563eb" : "#e5e7eb",
                  color: calibrationReady ? "#fff" : "#9ca3af",
                  display: "flex",
                  alignItems: "center",
                  justifyContent: "center",
                  gap: 8,
                  transition: "all 0.3s",
                }}
              >
                <Play style={{ width: 16, height: 16 }} />
                {calibrationReady ? "Begin Exam" : "Complete setup to continue"}
              </button>

              <button
                onClick={() => {
                  setCalibrationReady(true);
                  setSetupStep(2);
                  setQuizStarted(true);
                }}
                style={{
                  background: "none",
                  border: "none",
                  fontSize: 12,
                  color: "#9ca3af",
                  cursor: "pointer",
                  textDecoration: "underline",
                  textAlign: "center",
                }}
              >
                Skip camera setup (demo mode)
              </button>
            </div>
          )}

          {/* QUIZ QUESTIONS */}
          {quizStarted && (
            <div>
              <button
                onClick={handleBack}
                className="inline-flex items-center gap-2 text-gray-600 hover:text-blue-600 transition mb-6"
              >
                <ArrowLeft className="w-4 h-4" /> Back to Course
              </button>

              {isFrozen && (
                <div
                  style={{
                    background: "#fef2f2",
                    border: "1px solid #fca5a5",
                    borderRadius: 12,
                    padding: "12px 16px",
                    marginBottom: 16,
                    display: "flex",
                    alignItems: "center",
                    gap: 10,
                  }}
                >
                  <EyeOff className="w-5 h-5 text-red-500 flex-shrink-0" />
                  <div>
                    <p
                      style={{
                        fontSize: 14,
                        fontWeight: 700,
                        color: "#dc2626",
                        margin: 0,
                      }}
                    >
                      Exam Paused — Look at the Camera
                    </p>
                    <p style={{ fontSize: 12, color: "#b91c1c", margin: 0 }}>
                      Timer is frozen. Resume by facing the screen.
                    </p>
                  </div>
                </div>
              )}

              <div
                className="bg-white rounded-2xl border border-gray-100 shadow-sm p-6 mb-6"
                style={{
                  opacity: isFrozen ? 0.5 : 1,
                  pointerEvents: isFrozen ? "none" : "auto",
                  transition: "opacity 0.3s",
                  userSelect: isFrozen ? "none" : "auto",
                }}
              >
                <div className="flex items-center justify-between mb-4">
                  <div>
                    <p className="text-sm text-gray-500 mb-1">{title}</p>
                    <h1 className="text-xl font-bold text-gray-900">
                      Question {currentIdx + 1} of {questions.length}
                    </h1>
                  </div>
                  <div
                    className={`flex items-center gap-2 text-lg font-bold ${urgent ? "text-red-600" : "text-gray-900"}`}
                  >
                    <Clock
                      className={`w-5 h-5 ${urgent ? "animate-pulse text-red-600" : "text-blue-600"}`}
                    />
                    {isFrozen ? (
                      <span style={{ fontSize: 13, color: "#6b7280" }}>
                        Paused
                      </span>
                    ) : (
                      formatTime(timeLeft)
                    )}
                  </div>
                </div>
                <TimerBar timeLeft={timeLeft} totalTime={totalTime} />
                <div className="flex gap-2 mt-4 flex-wrap">
                  {questions.map((q, i) => (
                    <button
                      key={q.id}
                      onClick={() => setCurrentIdx(i)}
                      className={`w-8 h-8 rounded-lg text-xs font-semibold transition ${
                        i === currentIdx
                          ? "bg-blue-600 text-white"
                          : answers[q.id] !== undefined
                            ? "bg-green-100 text-green-700 border border-green-300"
                            : "bg-gray-100 text-gray-600 hover:bg-gray-200"
                      }`}
                    >
                      {i + 1}
                    </button>
                  ))}
                </div>
              </div>

              <div
                className="bg-white rounded-2xl border border-gray-100 shadow-sm p-6 mb-6"
                style={{
                  opacity: isFrozen ? 0.4 : 1,
                  pointerEvents: isFrozen ? "none" : "auto",
                  transition: "opacity 0.3s, filter 0.3s",
                  filter: isFrozen ? "blur(2px)" : "none",
                  userSelect: isFrozen ? "none" : "auto",
                }}
              >
                <p className="text-lg font-semibold text-gray-900 mb-6">
                  {currentQ.question_text}
                </p>
                <div className="space-y-3">
                  {currentQ.options.map((opt, i) => {
                    const selected = answers[currentQ.id] === opt.id;
                    const letter = String.fromCharCode(65 + i);
                    return (
                      <button
                        key={opt.id}
                        onClick={() =>
                          setAnswers((prev) => ({
                            ...prev,
                            [currentQ.id]: opt.id,
                          }))
                        }
                        className={`w-full text-left flex items-center gap-4 p-4 rounded-xl border-2 transition-all ${selected ? "border-blue-600 bg-blue-50" : "border-gray-200 bg-white hover:border-blue-300 hover:bg-blue-50/30"}`}
                      >
                        <span
                          className={`w-8 h-8 flex-shrink-0 flex items-center justify-center rounded-lg font-bold text-sm ${selected ? "bg-blue-600 text-white" : "bg-gray-100 text-gray-600"}`}
                        >
                          {letter}
                        </span>
                        <span className="text-gray-900 font-medium">
                          {opt.option_text}
                        </span>
                      </button>
                    );
                  })}
                </div>
              </div>

              <div
                className="flex items-center gap-4 mb-6"
                style={{
                  opacity: isFrozen ? 0.4 : 1,
                  pointerEvents: isFrozen ? "none" : "auto",
                  transition: "opacity 0.3s",
                }}
              >
                <button
                  onClick={() => setCurrentIdx((i) => Math.max(0, i - 1))}
                  disabled={currentIdx === 0}
                  className="px-4 py-2.5 border border-gray-200 text-gray-700 rounded-xl font-medium hover:bg-gray-50 disabled:opacity-50 disabled:cursor-not-allowed transition flex items-center gap-2"
                >
                  <ChevronLeft className="w-4 h-4" /> Prev
                </button>
                <div className="flex-1 text-center text-sm text-gray-600 font-medium">
                  {answeredCount}/{questions.length} answered
                </div>
                {isLast ? (
                  <button
                    onClick={handleSubmit}
                    className="px-5 py-2.5 bg-blue-600 text-white rounded-xl font-medium hover:bg-blue-700 transition flex items-center justify-center gap-2"
                  >
                    <CheckCircle className="w-4 h-4" /> Submit Quiz
                  </button>
                ) : (
                  <button
                    onClick={() =>
                      setCurrentIdx((i) =>
                        Math.min(questions.length - 1, i + 1),
                      )
                    }
                    className="px-4 py-2.5 border border-gray-200 text-gray-700 rounded-xl font-medium hover:bg-gray-50 transition flex items-center gap-2"
                  >
                    Next <ChevronRight className="w-4 h-4" />
                  </button>
                )}
              </div>

              {answeredCount === questions.length && !isLast && !isFrozen && (
                <div className="text-center">
                  <button
                    onClick={handleSubmit}
                    className="text-sm font-medium text-blue-600 hover:text-blue-700 underline transition"
                  >
                    All answered — submit early?
                  </button>
                </div>
              )}
            </div>
          )}
        </div>

        {/* ── RIGHT: iframe panel — always rendered, never conditionally wrapped ── */}
        <aside
          style={{
            width: 520,
            minWidth: 320,
            flexShrink: 0,
            position: "sticky",
            top: 24,
            alignSelf: "flex-start",
            display: "flex",
            flexDirection: "column",
            gap: 12,
          }}
        >
          <div
            style={{
              background: "#0f1117",
              borderRadius: 16,
              overflow: "hidden",
              border:
                isFrozen && quizStarted
                  ? "2px solid #ef4444"
                  : "2px solid #1e2230",
              boxShadow:
                isFrozen && quizStarted
                  ? "0 0 24px rgba(239,68,68,0.25)"
                  : "none",
              transition: "border-color 0.3s, box-shadow 0.3s",
            }}
          >
            {/* Header */}
            <div
              style={{
                display: "flex",
                alignItems: "center",
                justifyContent: "space-between",
                padding: "10px 14px",
                borderBottom: "1px solid #1e2230",
              }}
            >
              <div style={{ display: "flex", alignItems: "center", gap: 6 }}>
                <span
                  style={{
                    width: 8,
                    height: 8,
                    borderRadius: "50%",
                    background: sessionReady ? "#22c55e" : "#6b7280",
                    display: "inline-block",
                    boxShadow: sessionReady ? "0 0 6px #22c55e" : "none",
                    transition: "all 0.3s",
                  }}
                />
                <span
                  style={{
                    fontSize: 11,
                    fontWeight: 600,
                    color: "#9ca3af",
                    letterSpacing: "0.05em",
                    textTransform: "uppercase",
                  }}
                >
                  AI Proctoring
                </span>
              </div>
              {sessionReady ? (
                <ShieldCheck
                  style={{ width: 15, height: 15, color: "#22c55e" }}
                />
              ) : (
                <ShieldAlert
                  style={{ width: 15, height: 15, color: "#6b7280" }}
                />
              )}
            </div>

            {/* THE iframe — unconditionally rendered so React never remounts it */}
            <div style={{ position: "relative", background: "#070a0f" }}>
              {embedUrl ? (
                <iframe
                  src={embedUrl}
                  title="Proctoring"
                  allow="camera; microphone"
                  width="100%"
                  height={quizStarted ? "340" : "420"}
                  style={{
                    display: "block",
                    border: "none",
                    transition: "height 0.4s",
                  }}
                />
              ) : (
                <div
                  style={{
                    height: quizStarted ? 340 : 420,
                    display: "flex",
                    alignItems: "center",
                    justifyContent: "center",
                    flexDirection: "column",
                    gap: 8,
                    color: "#6b7280",
                  }}
                >
                  <Loader2
                    style={{
                      width: 28,
                      height: 28,
                      animation: "spin 1s linear infinite",
                    }}
                  />
                  <span style={{ fontSize: 13 }}>Connecting…</span>
                </div>
              )}

              {isFrozen && quizStarted && (
                <div
                  style={{
                    position: "absolute",
                    inset: 0,
                    background: "rgba(239,68,68,0.15)",
                    display: "flex",
                    alignItems: "center",
                    justifyContent: "center",
                    backdropFilter: "blur(1px)",
                  }}
                >
                  <div
                    style={{
                      background: "rgba(239,68,68,0.92)",
                      borderRadius: 8,
                      padding: "6px 14px",
                      fontSize: 11,
                      fontWeight: 700,
                      color: "#fff",
                      letterSpacing: "0.06em",
                      textTransform: "uppercase",
                      animation: "blink 1.4s ease-in-out infinite",
                    }}
                  >
                    ⚠ Look at the screen
                  </div>
                </div>
              )}
            </div>

            {/* Attention badge — shown during quiz only */}
            {quizStarted && (
              <div
                style={{
                  padding: "10px 14px",
                  borderTop: "1px solid #1e2230",
                  display: "flex",
                  alignItems: "center",
                  justifyContent: "space-between",
                }}
              >
                <span style={{ fontSize: 11, color: "#6b7280" }}>
                  Attention
                </span>
                <div
                  style={{
                    display: "flex",
                    alignItems: "center",
                    gap: 5,
                    fontSize: 12,
                    fontWeight: 600,
                    color: attentionColor,
                    transition: "color 0.3s",
                  }}
                >
                  {attentionState === "ATTENTIVE" && (
                    <Eye style={{ width: 13, height: 13 }} />
                  )}
                  {attentionState === "INATTENTIVE" && (
                    <EyeOff style={{ width: 13, height: 13 }} />
                  )}
                  {attentionState === "UNKNOWN" && (
                    <Loader2
                      style={{
                        width: 13,
                        height: 13,
                        animation: "spin 1s linear infinite",
                      }}
                    />
                  )}
                  {attentionLabel}
                </div>
              </div>
            )}
          </div>

          {/* Freeze warning — shown only when frozen during quiz */}
          {isFrozen && quizStarted && (
            <div
              style={{
                background: "#fef2f2",
                border: "1px solid #fecaca",
                borderRadius: 12,
                padding: "12px 14px",
              }}
            >
              <div
                style={{
                  display: "flex",
                  alignItems: "center",
                  gap: 6,
                  marginBottom: 4,
                }}
              >
                <EyeOff
                  style={{
                    width: 15,
                    height: 15,
                    color: "#dc2626",
                    flexShrink: 0,
                  }}
                />
                <span
                  style={{ fontSize: 13, fontWeight: 700, color: "#dc2626" }}
                >
                  Exam Paused
                </span>
              </div>
              <p
                style={{
                  fontSize: 12,
                  color: "#b91c1c",
                  margin: 0,
                  lineHeight: 1.5,
                }}
              >
                You looked away. Face the camera to resume automatically.
              </p>
            </div>
          )}
          {/* Integrity log removed */}
        </aside>
      </div>

      <style>{`
        @keyframes spin { to { transform: rotate(360deg); } }
        @keyframes blink { 0%,100%{opacity:1} 50%{opacity:0.6} }
      `}</style>
    </div>
  );
}
