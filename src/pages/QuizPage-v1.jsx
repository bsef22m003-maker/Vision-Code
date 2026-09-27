// import { useState, useEffect, useCallback, useRef } from "react";
// import { useLocation, useNavigate, useParams } from "react-router-dom";
// import {
//   ArrowLeft,
//   Clock,
//   CheckCircle,
//   XCircle,
//   Trophy,
//   AlertCircle,
//   ChevronRight,
//   ChevronLeft,
//   RotateCcw,
//   BarChart3,
//   Eye,
//   EyeOff,
//   ShieldCheck,
//   ShieldAlert,
//   WifiOff,
//   Loader2,
//   Camera,
//   Crosshair,
//   Play,
// } from "lucide-react";

// // ─── Config ───────────────────────────────────────────────────────────────────
// const BASE_URL = "https://examproctoringapi-production.up.railway.app";
// const POLL_INTERVAL_MS = 2000;

// const FREEZE_EVENTS = ["INATTENTIVE", "EYES_OFF_SCREEN"];

// // ─── Helpers ──────────────────────────────────────────────────────────────────
// function formatTime(seconds) {
//   const m = Math.floor(seconds / 60);
//   const s = seconds % 60;
//   return `${String(m).padStart(2, "0")}:${String(s).padStart(2, "0")}`;
// }

// function calcScore(questions, answers) {
//   return questions.reduce((acc, q) => {
//     const chosen = q.options.find((o) => o.id === answers[q.id]);
//     return acc + (chosen?.is_correct ? 1 : 0);
//   }, 0);
// }

// function isStateInattentive(currentState) {
//   if (!currentState) return false;
//   const attention = (currentState.attention ?? "").toUpperCase();
//   const status = (currentState.candidate_status ?? "").toUpperCase();
//   return (
//     attention === "INATTENTIVE" ||
//     status.includes("INATTENTIVE") ||
//     status.includes("EYES") ||
//     status.includes("AWAY") ||
//     status.includes("DEVIAT")
//   );
// }

// function parseSessionStatus(data) {
//   if (!data || typeof data !== "object")
//     return { faceRegistered: false, calibrated: false };

//   console.log("📊 Session poll response:", data);
//   const status = data.status ?? "";
//   const faceRegistered =
//     status === "calibrating" || status === "active" || status === "ended";
//   const calibrated = status === "active" || status === "ended";
//   return { faceRegistered, calibrated };
// }

// // ─── Timer Bar ────────────────────────────────────────────────────────────────
// function TimerBar({ timeLeft, totalTime }) {
//   const pct = Math.max(0, (timeLeft / totalTime) * 100);
//   const urgent = pct < 25;
//   const warning = pct < 50;
//   return (
//     <div className="w-full h-2 bg-gray-100 rounded-full overflow-hidden">
//       <div
//         className={`h-full rounded-full transition-all duration-1000 ${
//           urgent ? "bg-red-500" : warning ? "bg-yellow-500" : "bg-blue-600"
//         }`}
//         style={{ width: `${pct}%` }}
//       />
//     </div>
//   );
// }

// // ─── Setup Step Indicator ─────────────────────────────────────────────────────
// function SetupStep({ num, label, sublabel, status }) {
//   return (
//     <div style={{ display: "flex", alignItems: "center", gap: 12 }}>
//       <div
//         style={{
//           width: 36,
//           height: 36,
//           borderRadius: "50%",
//           flexShrink: 0,
//           display: "flex",
//           alignItems: "center",
//           justifyContent: "center",
//           fontWeight: 700,
//           fontSize: 14,
//           background:
//             status === "done"
//               ? "#dcfce7"
//               : status === "active"
//                 ? "#dbeafe"
//                 : "#f3f4f6",
//           border: `2px solid ${status === "done" ? "#22c55e" : status === "active" ? "#3b82f6" : "#e5e7eb"}`,
//           color:
//             status === "done"
//               ? "#16a34a"
//               : status === "active"
//                 ? "#2563eb"
//                 : "#9ca3af",
//           transition: "all 0.4s",
//         }}
//       >
//         {status === "done" ? (
//           <CheckCircle style={{ width: 18, height: 18 }} />
//         ) : (
//           num
//         )}
//       </div>
//       <div style={{ flex: 1 }}>
//         <div
//           style={{
//             fontSize: 13,
//             fontWeight: 600,
//             color: status === "waiting" ? "#9ca3af" : "#111827",
//           }}
//         >
//           {label}
//         </div>
//         <div style={{ fontSize: 11, color: "#9ca3af" }}>{sublabel}</div>
//       </div>
//       {status === "active" && (
//         <Loader2
//           style={{
//             width: 16,
//             height: 16,
//             color: "#3b82f6",
//             animation: "spin 1s linear infinite",
//             flexShrink: 0,
//           }}
//         />
//       )}
//       {status === "done" && (
//         <span
//           style={{
//             fontSize: 11,
//             color: "#16a34a",
//             fontWeight: 600,
//             flexShrink: 0,
//           }}
//         >
//           Done
//         </span>
//       )}
//     </div>
//   );
// }

// // ─── Result Screen ────────────────────────────────────────────────────────────
// function ResultScreen({
//   questions,
//   answers,
//   quizData,
//   title,
//   timeTaken,
//   totalTime,
//   onRetry,
//   onBack,
//   report,
// }) {
//   const score = calcScore(questions, answers);
//   const total = questions.length;
//   const passed = score >= quizData.passing_marks;
//   const pct = Math.round((score / total) * 100);

//   return (
//     <div className="min-h-screen bg-gray-50">
//       <div className="max-w-3xl mx-auto px-4 py-8">
//         <button
//           onClick={onBack}
//           className="inline-flex items-center gap-2 text-gray-600 hover:text-blue-600 transition mb-8"
//         >
//           <ArrowLeft className="w-4 h-4" /> Back to Course
//         </button>
//         <div className="bg-white rounded-2xl shadow-sm border border-gray-100 p-8 mb-8">
//           <div className="text-center mb-8">
//             <div
//               className={`w-20 h-20 mx-auto rounded-full flex items-center justify-center mb-6 ${passed ? "bg-green-100" : "bg-red-100"}`}
//             >
//               {passed ? (
//                 <Trophy className="w-10 h-10 text-green-600" />
//               ) : (
//                 <XCircle className="w-10 h-10 text-red-600" />
//               )}
//             </div>
//             <h1
//               className={`text-3xl font-bold mb-2 ${passed ? "text-green-700" : "text-red-700"}`}
//             >
//               {passed ? "Quiz Passed! 🎉" : "Quiz Failed"}
//             </h1>
//             <p className="text-gray-600 text-sm">{title}</p>
//           </div>
//           <div className="bg-gradient-to-r from-blue-50 to-purple-50 rounded-xl p-8 mb-8">
//             <div className="text-center mb-6">
//               <div className="flex items-center justify-center gap-2 mb-4">
//                 <span className="text-6xl font-bold text-gray-900">
//                   {score}
//                 </span>
//                 <span className="text-2xl text-gray-500">/ {total}</span>
//               </div>
//               <div className="text-3xl font-bold text-blue-600">{pct}%</div>
//             </div>
//             <div className="w-full h-3 bg-gray-200 rounded-full overflow-hidden mb-6">
//               <div
//                 className={`h-full rounded-full transition-all duration-700 ${passed ? "bg-green-500" : "bg-red-500"}`}
//                 style={{ width: `${pct}%` }}
//               />
//             </div>
//             <div className="grid grid-cols-3 gap-4 text-sm">
//               {[
//                 [
//                   "Passing Score",
//                   `${Math.round((quizData.passing_marks / total) * 100)}%`,
//                 ],
//                 ["Your Score", `${pct}%`],
//                 ["Time Taken", formatTime(timeTaken)],
//               ].map(([label, val]) => (
//                 <div
//                   key={label}
//                   className="bg-white rounded-lg p-3 text-center"
//                 >
//                   <p className="text-gray-500 text-xs mb-1">{label}</p>
//                   <p className="font-bold text-gray-900">{val}</p>
//                 </div>
//               ))}
//             </div>
//           </div>
//           {report?.integrity_verdict && (
//             <div className="mb-8 border border-gray-200 rounded-xl overflow-hidden">
//               <div className="bg-gray-50 px-4 py-3 border-b border-gray-200 flex items-center gap-2">
//                 <ShieldCheck className="w-4 h-4 text-blue-600" />
//                 <span className="text-sm font-semibold text-gray-800">
//                   Proctoring Report
//                 </span>
//               </div>
//               <pre className="p-4 text-xs text-gray-700 overflow-x-auto bg-white">
//                 {JSON.stringify(report.integrity_verdict, null, 2)}
//               </pre>
//             </div>
//           )}
//           <div className="mb-8">
//             <h2 className="text-xl font-bold text-gray-900 mb-4 flex items-center gap-2">
//               <BarChart3 className="w-5 h-5 text-blue-600" /> Answer Review
//             </h2>
//             <div className="space-y-3">
//               {questions.map((q, idx) => {
//                 const chosen = q.options.find((o) => o.id === answers[q.id]);
//                 const isRight = chosen?.is_correct ?? false;
//                 return (
//                   <div
//                     key={q.id}
//                     className={`border rounded-lg p-4 ${isRight ? "bg-green-50 border-green-200" : "bg-red-50 border-red-200"}`}
//                   >
//                     <div className="flex items-start gap-3 mb-3">
//                       {isRight ? (
//                         <CheckCircle className="w-5 h-5 text-green-600 flex-shrink-0 mt-0.5" />
//                       ) : (
//                         <XCircle className="w-5 h-5 text-red-600 flex-shrink-0 mt-0.5" />
//                       )}
//                       <p className="font-semibold text-gray-900 text-sm">
//                         Q{idx + 1}. {q.question_text}
//                       </p>
//                     </div>
//                     <div className="space-y-2 ml-8">
//                       {q.options.map((opt) => {
//                         const isChosen = opt.id === answers[q.id];
//                         const isCorrect = opt.is_correct;
//                         let cls = "bg-white border-gray-200 text-gray-700";
//                         if (isCorrect)
//                           cls = "bg-green-100 border-green-300 text-green-800";
//                         else if (isChosen && !isCorrect)
//                           cls = "bg-red-100 border-red-300 text-red-800";
//                         return (
//                           <div
//                             key={opt.id}
//                             className={`text-xs px-3 py-2 rounded border flex items-center gap-2 ${cls}`}
//                           >
//                             {isCorrect && (
//                               <CheckCircle className="w-3 h-3 text-green-600 flex-shrink-0" />
//                             )}
//                             {isChosen && !isCorrect && (
//                               <XCircle className="w-3 h-3 text-red-600 flex-shrink-0" />
//                             )}
//                             {!isCorrect && !isChosen && (
//                               <span className="w-3 h-3 flex-shrink-0" />
//                             )}
//                             {opt.option_text}
//                           </div>
//                         );
//                       })}
//                       {!chosen && (
//                         <p className="text-xs text-yellow-700 flex items-center gap-1 mt-2">
//                           <AlertCircle className="w-3 h-3" /> Not answered
//                         </p>
//                       )}
//                     </div>
//                   </div>
//                 );
//               })}
//             </div>
//           </div>
//           <div className="flex gap-3">
//             <button
//               onClick={onBack}
//               className="flex-1 px-4 py-3 border border-gray-200 rounded-xl font-semibold text-gray-700 hover:bg-gray-50 transition flex items-center justify-center gap-2"
//             >
//               <ArrowLeft className="w-4 h-4" /> Back to Course
//             </button>
//             <button
//               onClick={onRetry}
//               className="flex-1 px-4 py-3 bg-blue-600 text-white rounded-xl font-semibold hover:bg-blue-700 transition flex items-center justify-center gap-2"
//             >
//               <RotateCcw className="w-4 h-4" /> Retry Quiz
//             </button>
//           </div>
//         </div>
//       </div>
//     </div>
//   );
// }

// // ─── Proctoring Sidebar (active quiz) — no iframe, reads from props ───────────
// function ProctoringPanel({ isFrozen, alerts, attentionState, sessionReady }) {
//   const stateColor =
//     attentionState === "ATTENTIVE"
//       ? "#16a34a"
//       : attentionState === "INATTENTIVE"
//         ? "#dc2626"
//         : "#9ca3af";

//   const stateLabel =
//     attentionState === "ATTENTIVE"
//       ? "Attentive"
//       : attentionState === "INATTENTIVE"
//         ? "Inattentive"
//         : "Monitoring…";

//   return (
//     <aside
//       style={{
//         width: 280,
//         minWidth: 280,
//         maxWidth: 280,
//         display: "flex",
//         flexDirection: "column",
//         gap: 12,
//         position: "sticky",
//         top: 24,
//         alignSelf: "flex-start",
//       }}
//     >
//       {/* Camera feed placeholder — the REAL iframe is rendered persistently in the root */}
//       <div
//         style={{
//           background: "#0f1117",
//           borderRadius: 16,
//           overflow: "hidden",
//           border: isFrozen ? "2px solid #ef4444" : "2px solid #1e2230",
//           boxShadow: isFrozen ? "0 0 24px rgba(239,68,68,0.25)" : "none",
//           transition: "border-color 0.3s, box-shadow 0.3s",
//         }}
//       >
//         {/* Header */}
//         <div
//           style={{
//             display: "flex",
//             alignItems: "center",
//             justifyContent: "space-between",
//             padding: "10px 14px",
//             borderBottom: "1px solid #1e2230",
//           }}
//         >
//           <div style={{ display: "flex", alignItems: "center", gap: 6 }}>
//             <span
//               style={{
//                 width: 8,
//                 height: 8,
//                 borderRadius: "50%",
//                 background: sessionReady ? "#22c55e" : "#6b7280",
//                 display: "inline-block",
//                 boxShadow: sessionReady ? "0 0 6px #22c55e" : "none",
//                 transition: "all 0.3s",
//               }}
//             />
//             <span
//               style={{
//                 fontSize: 11,
//                 fontWeight: 600,
//                 color: "#9ca3af",
//                 letterSpacing: "0.05em",
//                 textTransform: "uppercase",
//               }}
//             >
//               AI Proctoring
//             </span>
//           </div>
//           {sessionReady ? (
//             <ShieldCheck style={{ width: 15, height: 15, color: "#22c55e" }} />
//           ) : (
//             <ShieldAlert style={{ width: 15, height: 15, color: "#6b7280" }} />
//           )}
//         </div>

//         {/*
//           The iframe is NOT rendered here. Instead we use a placeholder div
//           that the persistent iframe (rendered at root level) visually overlaps.
//           We just reserve the correct height so the layout doesn't jump.
//         */}
//         <div
//           id="proctoring-iframe-slot"
//           style={{
//             position: "relative",
//             background: "#070a0f",
//             height: 200,
//           }}
//         >
//           {/* Frozen overlay */}
//           {isFrozen && (
//             <div
//               style={{
//                 position: "absolute",
//                 inset: 0,
//                 background: "rgba(239,68,68,0.15)",
//                 display: "flex",
//                 alignItems: "center",
//                 justifyContent: "center",
//                 backdropFilter: "blur(1px)",
//                 zIndex: 10,
//               }}
//             >
//               <div
//                 style={{
//                   background: "rgba(239,68,68,0.92)",
//                   borderRadius: 8,
//                   padding: "6px 14px",
//                   fontSize: 11,
//                   fontWeight: 700,
//                   color: "#fff",
//                   letterSpacing: "0.06em",
//                   textTransform: "uppercase",
//                   animation: "blink 1.4s ease-in-out infinite",
//                 }}
//               >
//                 ⚠ Look at the screen
//               </div>
//             </div>
//           )}
//         </div>

//         {/* Attention status */}
//         <div
//           style={{
//             padding: "10px 14px",
//             borderTop: "1px solid #1e2230",
//             display: "flex",
//             alignItems: "center",
//             justifyContent: "space-between",
//           }}
//         >
//           <span style={{ fontSize: 11, color: "#6b7280" }}>Attention</span>
//           <div
//             style={{
//               display: "flex",
//               alignItems: "center",
//               gap: 5,
//               fontSize: 12,
//               fontWeight: 600,
//               color: stateColor,
//               transition: "color 0.3s",
//             }}
//           >
//             {attentionState === "ATTENTIVE" && (
//               <Eye style={{ width: 13, height: 13 }} />
//             )}
//             {attentionState === "INATTENTIVE" && (
//               <EyeOff style={{ width: 13, height: 13 }} />
//             )}
//             {attentionState === "UNKNOWN" && (
//               <Loader2
//                 style={{
//                   width: 13,
//                   height: 13,
//                   animation: "spin 1s linear infinite",
//                 }}
//               />
//             )}
//             {stateLabel}
//           </div>
//         </div>
//       </div>

//       {/* Freeze warning */}
//       {isFrozen && (
//         <div
//           style={{
//             background: "#fef2f2",
//             border: "1px solid #fecaca",
//             borderRadius: 12,
//             padding: "12px 14px",
//           }}
//         >
//           <div
//             style={{
//               display: "flex",
//               alignItems: "center",
//               gap: 6,
//               marginBottom: 4,
//             }}
//           >
//             <EyeOff
//               style={{ width: 15, height: 15, color: "#dc2626", flexShrink: 0 }}
//             />
//             <span style={{ fontSize: 13, fontWeight: 700, color: "#dc2626" }}>
//               Exam Paused
//             </span>
//           </div>
//           <p
//             style={{
//               fontSize: 12,
//               color: "#b91c1c",
//               margin: 0,
//               lineHeight: 1.5,
//             }}
//           >
//             You looked away. Face the camera to resume automatically.
//           </p>
//         </div>
//       )}

//       {/* Integrity log */}
//       <div
//         style={{
//           background: "#fff",
//           border: "1px solid #e5e7eb",
//           borderRadius: 12,
//           overflow: "hidden",
//         }}
//       >
//         <div
//           style={{
//             padding: "10px 14px",
//             borderBottom: "1px solid #f3f4f6",
//             display: "flex",
//             alignItems: "center",
//             justifyContent: "space-between",
//           }}
//         >
//           <span style={{ fontSize: 12, fontWeight: 600, color: "#374151" }}>
//             Integrity Log
//           </span>
//           {alerts.length > 0 && (
//             <span
//               style={{
//                 background: "#fee2e2",
//                 color: "#dc2626",
//                 fontSize: 10,
//                 fontWeight: 700,
//                 padding: "2px 7px",
//                 borderRadius: 99,
//               }}
//             >
//               {alerts.length}
//             </span>
//           )}
//         </div>
//         <div
//           style={{
//             maxHeight: 140,
//             overflowY: "auto",
//             padding: alerts.length === 0 ? "16px 14px" : "8px 0",
//           }}
//         >
//           {alerts.length === 0 ? (
//             <div style={{ textAlign: "center" }}>
//               <CheckCircle
//                 style={{
//                   width: 18,
//                   height: 18,
//                   margin: "0 auto 4px",
//                   color: "#22c55e",
//                 }}
//               />
//               <p style={{ fontSize: 12, color: "#9ca3af", margin: 0 }}>
//                 No alerts yet
//               </p>
//             </div>
//           ) : (
//             [...alerts].reverse().map((a, i) => (
//               <div
//                 key={i}
//                 style={{
//                   padding: "7px 14px",
//                   borderBottom:
//                     i < alerts.length - 1 ? "1px solid #f9fafb" : "none",
//                   display: "flex",
//                   gap: 8,
//                   alignItems: "flex-start",
//                 }}
//               >
//                 <span
//                   style={{
//                     width: 6,
//                     height: 6,
//                     borderRadius: "50%",
//                     marginTop: 5,
//                     flexShrink: 0,
//                     background:
//                       a.severity === "HIGH" || a.severity === "critical"
//                         ? "#ef4444"
//                         : a.severity === "MEDIUM" || a.severity === "warning"
//                           ? "#f59e0b"
//                           : "#6b7280",
//                   }}
//                 />
//                 <div>
//                   <p
//                     style={{
//                       fontSize: 11,
//                       fontWeight: 600,
//                       color: "#374151",
//                       margin: 0,
//                     }}
//                   >
//                     {a.event}
//                   </p>
//                   <p style={{ fontSize: 10, color: "#9ca3af", margin: 0 }}>
//                     {a.message}
//                   </p>
//                 </div>
//               </div>
//             ))
//           )}
//         </div>
//       </div>

//       <style>{`
//         @keyframes spin { to { transform: rotate(360deg); } }
//         @keyframes blink { 0%,100%{opacity:1} 50%{opacity:0.6} }
//       `}</style>
//     </aside>
//   );
// }

// // ─── Persistent Iframe — rendered once, never unmounted ───────────────────────
// // During "setup": visible inline inside the setup layout (position:static).
// // During "quiz":  absolutely positioned to overlap the #proctoring-iframe-slot div.
// // This keeps the same iframe alive across screen transitions so calibration
// // data, camera stream, and postMessage channel all persist.
// function PersistentProctoringIframe({ sessionId, screen }) {
//   const iframeRef = useRef(null);
//   const slotRef = useRef(null);
//   const rafRef = useRef(null);

//   const embedUrl = sessionId
//     ? `${BASE_URL}/?session_id=${sessionId}`
//     : null;

//   // During quiz mode, pin the iframe over the slot element every animation frame.
//   useEffect(() => {
//     if (screen !== "quiz") {
//       if (rafRef.current) cancelAnimationFrame(rafRef.current);
//       return;
//     }

//     const pin = () => {
//       const slot = document.getElementById("proctoring-iframe-slot");
//       const iframe = iframeRef.current;
//       if (slot && iframe) {
//         const rect = slot.getBoundingClientRect();
//         iframe.style.position = "fixed";
//         iframe.style.top = `${rect.top}px`;
//         iframe.style.left = `${rect.left}px`;
//         iframe.style.width = `${rect.width}px`;
//         iframe.style.height = `${rect.height}px`;
//         iframe.style.zIndex = "5";
//         iframe.style.borderRadius = "0";
//       }
//       rafRef.current = requestAnimationFrame(pin);
//     };

//     rafRef.current = requestAnimationFrame(pin);
//     return () => {
//       if (rafRef.current) cancelAnimationFrame(rafRef.current);
//     };
//   }, [screen]);

//   if (!embedUrl) return null;

//   // Setup mode: render normally as a block element inside the setup layout.
//   // Quiz mode: render off-screen initially; the RAF loop will pin it to the slot.
//   const setupStyle = {
//     display: "block",
//     border: "none",
//     width: "100%",
//     height: "480px",
//   };

//   const quizStyle = {
//     position: "fixed",
//     top: 0,
//     left: 0,
//     width: "280px",
//     height: "200px",
//     border: "none",
//     zIndex: 5,
//     display: "block",
//   };

//   return (
//     <iframe
//       ref={iframeRef}
//       src={embedUrl}
//       title="Proctoring"
//       allow="camera; microphone"
//       style={screen === "quiz" ? quizStyle : setupStyle}
//     />
//   );
// }

// // ─── Main Quiz Page ───────────────────────────────────────────────────────────
// export default function QuizPage() {
//   const { quizItemId } = useParams();
//   const location = useLocation();
//   const navigate = useNavigate();

//   const locationState = location.state;
//   const quizData = locationState?.quizData ?? null;
//   const title = locationState?.title ?? "Quiz";
//   const courseId = locationState?.courseId;
//   const totalTime = (locationState?.estimatedMinutes ?? 10) * 60;

//   // ── Quiz state ─────────────────────────────────────────────────────────────
//   const [currentIdx, setCurrentIdx] = useState(0);
//   const [answers, setAnswers] = useState({});
//   const [submitted, setSubmitted] = useState(false);
//   const [timeLeft, setTimeLeft] = useState(totalTime);
//   const [timeTaken, setTimeTaken] = useState(0);
//   const [quizStarted, setQuizStarted] = useState(false);
//   const timerRef = useRef(null);
//   const questions = quizData?.questions ?? [];

//   // ── Proctoring state ───────────────────────────────────────────────────────
//   const [sessionId, setSessionId] = useState(null);
//   const [sessionReady, setSessionReady] = useState(false);
//   const [alerts, setAlerts] = useState([]);
//   const [report, setReport] = useState(null);
//   const [attentionState, setAttentionState] = useState("UNKNOWN");
//   const [isFrozen, setIsFrozen] = useState(false);
//   const hasEndedRef = useRef(false);

//   // ── Screen flow: "info" → "setup" → "quiz" ────────────────────────────────
//   const [screen, setScreen] = useState("info");
//   const [setupStep, setSetupStep] = useState(0);
//   const setupStepRef = useRef(0);
//   const [calibrationReady, setCalibrationReady] = useState(false);

//   // Ref mirrors so closures always read fresh values
//   const screenRef = useRef(screen);
//   const quizStartedRef = useRef(quizStarted);
//   useEffect(() => { screenRef.current = screen; }, [screen]);
//   useEffect(() => { quizStartedRef.current = quizStarted; }, [quizStarted]);
//   useEffect(() => { setupStepRef.current = setupStep; }, [setupStep]);

//   // ── Create session when entering setup screen ──────────────────────────────
//   useEffect(() => {
//     if (screen !== "setup") return;
//     const createSession = async () => {
//       try {
//         const res = await fetch(`${BASE_URL}/api/sessions`, {
//           method: "POST",
//           headers: { "Content-Type": "application/json" },
//           body: JSON.stringify({
//             candidate_id: "cand_001",
//             candidate_name: "Test User",
//             exam_id: "exam_001",
//             exam_label: title,
//           }),
//         });
//         const data = await res.json();
//         setSessionId(data.session_id);
//         setSessionReady(true);
//       } catch (err) {
//         console.error("❌ Session creation failed:", err);
//       }
//     };
//     createSession();
//   }, [screen, title]);

//   // ── Poll session status during setup ──────────────────────────────────────
//   useEffect(() => {
//     if (!sessionId || screen !== "setup") return;

//     const poll = async () => {
//       try {
//         const res = await fetch(`${BASE_URL}/api/sessions/${sessionId}`);
//         if (!res.ok) return;
//         const data = await res.json();
//         const { faceRegistered, calibrated } = parseSessionStatus(data);
//         if (faceRegistered && setupStepRef.current === 0) {
//           setSetupStep(1);
//         }
//         if (calibrated && setupStepRef.current >= 1) {
//           setSetupStep(2);
//           setCalibrationReady(true);
//         }
//         const attention = data.current_state?.attention;
//         if (attention) setAttentionState(attention);
//       } catch {
//         // fail silently
//       }
//     };

//     poll();
//     const intervalId = setInterval(poll, POLL_INTERVAL_MS);
//     return () => clearInterval(intervalId);
//   }, [sessionId, screen]);

//   // ── Single postMessage listener — source of truth for attention ───────────
//   // This fires from the PERSISTENT iframe, so it works in both setup and quiz.
//   useEffect(() => {
//     const allowedOrigin = (() => {
//       try { return new URL(BASE_URL).origin; } catch { return BASE_URL; }
//     })();

//     const handler = (event) => {
//       if (!event.data || typeof event.data !== "object") return;
//       if (event.origin !== allowedOrigin) return;

//       const data = event.data;
//       const type = (data.type ?? "").toLowerCase();
//       const currentScreen = screenRef.current;
//       const currentStep = setupStepRef.current;

//       console.log("📨 postMessage:", type, data);

//       // ── Setup step progression ─────────────────────────────────────────
//       if (currentScreen === "setup") {
//         const isFaceEvent =
//           type.includes("face") ||
//           type.includes("registr") ||
//           type === "step_1_complete" ||
//           type === "step1_complete";

//         if (isFaceEvent && currentStep === 0) {
//           setSetupStep(1);
//         }

//         const isCalibrationEvent =
//           type.includes("calibrat") ||
//           type.includes("setup_complete") ||
//           type === "ready" ||
//           type === "gaze_calibrated" ||
//           type === "step_2_complete" ||
//           type === "step2_complete" ||
//           type === "complete";

//         if (isCalibrationEvent && currentStep >= 1) {
//           setSetupStep(2);
//           setCalibrationReady(true);
//         }

//         if (
//           currentStep === 0 &&
//           !isFaceEvent &&
//           (type.includes("complet") ||
//             type.includes("done") ||
//             type.includes("success") ||
//             type.includes("ready"))
//         ) {
//           setSetupStep(1);
//         }
//       }

//       // ── Metrics — this is the main attention signal ────────────────────
//       // Fires continuously from the iframe; we trust it over API polling.
//       if (type === "metrics") {
//         const cs = data.state ?? data.current_state ?? {};
//         const inattentive = isStateInattentive(cs);
//         const attention = (cs.attention ?? "").toUpperCase();

//         // During setup: a valid attention reading means calibration is done
//         if (currentScreen === "setup" && attention && attention !== "UNKNOWN") {
//           setSetupStep(2);
//           setCalibrationReady(true);
//         }

//         // Always update the attention badge
//         setAttentionState(inattentive ? "INATTENTIVE" : attention || "UNKNOWN");

//         // Only freeze/unfreeze during active quiz
//         if (quizStartedRef.current) {
//           if (inattentive) {
//             setIsFrozen(true);
//           } else if (attention === "ATTENTIVE") {
//             setIsFrozen(false);
//           }
//         }
//       }

//       // ── Alert events ───────────────────────────────────────────────────
//       if (type === "alert" || type === "integrity_alert") {
//         setAlerts((prev) => [...prev, data]);
//         if (quizStartedRef.current && FREEZE_EVENTS.includes(data.event)) {
//           setIsFrozen(true);
//           setAttentionState("INATTENTIVE");
//         }
//       }

//       if (type === "session_ended" || type === "session_complete") {
//         setReport(data.report ?? null);
//       }
//     };

//     window.addEventListener("message", handler);
//     return () => window.removeEventListener("message", handler);
//   }, []); // Intentionally empty — refs keep values fresh without re-subscribing

//   // ── Poll attention during active quiz (fallback only) ─────────────────────
//   // postMessage is the primary signal. This poll is a safety net for cases
//   // where the iframe doesn't emit metrics events frequently enough.
//   useEffect(() => {
//     if (!sessionId || !quizStarted || submitted) return;

//     const poll = async () => {
//       try {
//         const res = await fetch(`${BASE_URL}/api/sessions/${sessionId}`);
//         if (!res.ok) return;
//         const data = await res.json();

//         const currentState = data.current_state ?? {};
//         const attention = (currentState.attention ?? "").toUpperCase();
//         const inattentive = isStateInattentive(currentState);

//         // Only update if postMessage hasn't already given a fresh signal.
//         // We do this by checking if we're currently UNKNOWN (no signal yet).
//         // Once postMessage fires, it takes over as the primary signal.
//         if (attentionState === "UNKNOWN" && attention) {
//           setAttentionState(inattentive ? "INATTENTIVE" : "ATTENTIVE");
//         }

//         // Check for new freeze-worthy alerts from recent_alerts
//         if (data.recent_alerts && Array.isArray(data.recent_alerts)) {
//           setAlerts((prev) => {
//             const existing = new Set(prev.map((a) => JSON.stringify(a)));
//             const incoming = data.recent_alerts.filter(
//               (a) => !existing.has(JSON.stringify(a)),
//             );
//             if (incoming.some((a) => FREEZE_EVENTS.includes(a.event))) {
//               setIsFrozen(true);
//               setAttentionState("INATTENTIVE");
//             }
//             return incoming.length ? [...prev, ...incoming] : prev;
//           });
//         }
//       } catch {
//         // fail silently
//       }
//     };

//     poll();
//     const intervalId = setInterval(poll, POLL_INTERVAL_MS);
//     return () => clearInterval(intervalId);
//   }, [sessionId, quizStarted, submitted]); // eslint-disable-line react-hooks/exhaustive-deps

//   // ── End proctoring session ─────────────────────────────────────────────────
//   const endProctoringSession = useCallback(async () => {
//     if (!sessionId || hasEndedRef.current) return;
//     hasEndedRef.current = true;
//     try {
//       await fetch(`${BASE_URL}/api/sessions/${sessionId}/end`, {
//         method: "POST",
//       });
//       const res = await fetch(`${BASE_URL}/api/sessions/${sessionId}/report`);
//       const data = await res.json();
//       setReport(data);
//     } catch (err) {
//       console.error("❌ End session error:", err);
//     }
//   }, [sessionId]);

//   // ── Timer — pauses when frozen ─────────────────────────────────────────────
//   const stopTimer = useCallback(() => {
//     if (timerRef.current) clearInterval(timerRef.current);
//   }, []);

//   useEffect(() => {
//     if (!quizStarted || submitted || isFrozen) {
//       stopTimer();
//       return;
//     }
//     timerRef.current = setInterval(() => {
//       setTimeLeft((t) => {
//         if (t <= 1) {
//           stopTimer();
//           setTimeTaken(totalTime);
//           setSubmitted(true);
//           return 0;
//         }
//         return t - 1;
//       });
//     }, 1000);
//     return stopTimer;
//   }, [quizStarted, submitted, isFrozen, stopTimer, totalTime]);

//   // ── Handlers ───────────────────────────────────────────────────────────────
//   const handleSubmit = () => {
//     stopTimer();
//     setTimeTaken(totalTime - timeLeft);
//     setSubmitted(true);
//     endProctoringSession();
//   };

//   const handleRetry = () => {
//     hasEndedRef.current = false;
//     setAnswers({});
//     setSubmitted(false);
//     setCurrentIdx(0);
//     setTimeLeft(totalTime);
//     setTimeTaken(0);
//     setQuizStarted(false);
//     setIsFrozen(false);
//     setAlerts([]);
//     setReport(null);
//     setAttentionState("UNKNOWN");
//     setSetupStep(0);
//     setCalibrationReady(false);
//     setSessionId(null);
//     setSessionReady(false);
//     setScreen("setup");
//   };

//   const handleBack = () =>
//     navigate(courseId ? `/student/course/${courseId}` : "/student");

//   // ── Guard ──────────────────────────────────────────────────────────────────
//   if (!quizData) {
//     return (
//       <div className="min-h-screen bg-gray-50 flex items-center justify-center">
//         <div className="bg-white rounded-2xl shadow-sm border border-gray-100 p-8 text-center max-w-md">
//           <AlertCircle className="w-12 h-12 text-gray-400 mx-auto mb-4" />
//           <p className="text-lg font-bold text-gray-900 mb-4">
//             Quiz data not found.
//           </p>
//           <button
//             onClick={() => navigate(-1)}
//             className="w-full px-4 py-3 bg-blue-600 text-white rounded-xl font-semibold hover:bg-blue-700 transition"
//           >
//             Go Back
//           </button>
//         </div>
//       </div>
//     );
//   }

//   if (submitted) {
//     return (
//       <ResultScreen
//         questions={questions}
//         answers={answers}
//         quizData={quizData}
//         title={title}
//         timeTaken={timeTaken}
//         totalTime={totalTime}
//         onRetry={handleRetry}
//         onBack={handleBack}
//         report={report}
//       />
//     );
//   }

//   // ── Info screen ────────────────────────────────────────────────────────────
//   if (screen === "info") {
//     return (
//       <div className="min-h-screen bg-gray-50 flex items-center justify-center px-4">
//         <div className="bg-white rounded-2xl shadow-sm border border-gray-100 p-8 max-w-md w-full">
//           <div className="w-16 h-16 mx-auto bg-blue-100 rounded-full flex items-center justify-center mb-6">
//             <Clock className="w-8 h-8 text-blue-600" />
//           </div>
//           <h1 className="text-2xl font-bold text-gray-900 text-center mb-2">
//             {title}
//           </h1>
//           <p className="text-gray-600 text-sm text-center mb-6">
//             Ready to test your knowledge?
//           </p>
//           <div className="bg-blue-50 border border-blue-100 rounded-xl p-4 mb-6 flex items-start gap-3">
//             <ShieldCheck className="w-5 h-5 text-blue-600 flex-shrink-0 mt-0.5" />
//             <div>
//               <p className="text-sm font-semibold text-blue-800 mb-1">
//                 AI Proctoring Enabled
//               </p>
//               <p className="text-xs text-blue-700 leading-relaxed">
//                 You'll set up your camera before the exam begins. Looking away
//                 will <strong>pause</strong> the exam and timer until you face
//                 the screen again.
//               </p>
//             </div>
//           </div>
//           <div className="space-y-3 mb-8">
//             {[
//               ["Questions", questions.length],
//               ["Time Limit", formatTime(totalTime)],
//               [
//                 "Passing Score",
//                 `${quizData.passing_marks}/${quizData.total_marks}`,
//               ],
//             ].map(([label, val]) => (
//               <div
//                 key={label}
//                 className="bg-gray-50 rounded-lg p-4 flex items-center justify-between"
//               >
//                 <span className="text-sm text-gray-600 font-medium">
//                   {label}
//                 </span>
//                 <span className="text-xl font-bold text-blue-600">{val}</span>
//               </div>
//             ))}
//           </div>
//           <button
//             onClick={() => setScreen("setup")}
//             className="w-full px-4 py-3 bg-blue-600 text-white rounded-xl font-semibold hover:bg-blue-700 transition mb-3 flex items-center justify-center gap-2"
//           >
//             <Camera className="w-4 h-4" /> Set Up Camera & Start
//           </button>
//           <button
//             onClick={handleBack}
//             className="w-full px-4 py-3 border border-gray-200 text-gray-700 rounded-xl font-semibold hover:bg-gray-50 transition"
//           >
//             Cancel
//           </button>
//         </div>
//       </div>
//     );
//   }

//   // ── Setup screen ───────────────────────────────────────────────────────────
//   if (screen === "setup") {
//     const step0Status =
//       setupStep === 0 ? "active" : setupStep > 0 ? "done" : "waiting";
//     const step1Status =
//       setupStep === 1 ? "active" : setupStep > 1 ? "done" : "waiting";

//     return (
//       <div
//         style={{
//           minHeight: "100vh",
//           background: "#f9fafb",
//           display: "flex",
//           alignItems: "center",
//           justifyContent: "center",
//           padding: 24,
//         }}
//       >
//         <div
//           style={{
//             maxWidth: 860,
//             width: "100%",
//             display: "flex",
//             gap: 24,
//             alignItems: "flex-start",
//           }}
//         >
//           {/* Left: persistent iframe renders here in setup mode */}
//           <div
//             style={{
//               flex: 1,
//               background: "#0f1117",
//               borderRadius: 16,
//               overflow: "hidden",
//               border: "2px solid #1e2230",
//               minHeight: 480,
//             }}
//           >
//             {sessionId ? (
//               <PersistentProctoringIframe sessionId={sessionId} screen="setup" />
//             ) : (
//               <div
//                 style={{
//                   height: 480,
//                   display: "flex",
//                   alignItems: "center",
//                   justifyContent: "center",
//                   flexDirection: "column",
//                   gap: 8,
//                   color: "#6b7280",
//                 }}
//               >
//                 <Loader2
//                   style={{
//                     width: 28,
//                     height: 28,
//                     animation: "spin 1s linear infinite",
//                   }}
//                 />
//                 <span style={{ fontSize: 13 }}>
//                   Connecting to proctoring service…
//                 </span>
//               </div>
//             )}
//           </div>

//           {/* Right: instructions panel */}
//           <div
//             style={{
//               width: 280,
//               display: "flex",
//               flexDirection: "column",
//               gap: 14,
//             }}
//           >
//             <div
//               style={{
//                 background: "#fff",
//                 borderRadius: 14,
//                 border: "1px solid #e5e7eb",
//                 padding: "18px 20px",
//               }}
//             >
//               <div
//                 style={{
//                   display: "flex",
//                   alignItems: "center",
//                   gap: 8,
//                   marginBottom: 6,
//                 }}
//               >
//                 <ShieldCheck
//                   style={{ width: 18, height: 18, color: "#2563eb" }}
//                 />
//                 <span
//                   style={{ fontWeight: 700, fontSize: 14, color: "#111827" }}
//                 >
//                   Camera Setup Required
//                 </span>
//               </div>
//               <p
//                 style={{
//                   fontSize: 12,
//                   color: "#6b7280",
//                   margin: 0,
//                   lineHeight: 1.6,
//                 }}
//               >
//                 Complete both steps in the camera panel on the left. The quiz
//                 will unlock automatically once done.
//               </p>
//             </div>

//             <div
//               style={{
//                 background: "#fff",
//                 borderRadius: 14,
//                 border: "1px solid #e5e7eb",
//                 padding: "18px 20px",
//                 display: "flex",
//                 flexDirection: "column",
//                 gap: 16,
//               }}
//             >
//               <SetupStep
//                 num={1}
//                 label="Face Registration"
//                 sublabel='Center face → click "Capture Face"'
//                 status={step0Status}
//               />
//               <div style={{ height: 1, background: "#f3f4f6" }} />
//               <SetupStep
//                 num={2}
//                 label="Gaze Calibration"
//                 sublabel='Click "Begin" → follow the dot'
//                 status={step1Status}
//               />
//             </div>

//             <div
//               style={{
//                 background:
//                   setupStep === 0
//                     ? "#eff6ff"
//                     : setupStep === 1
//                       ? "#fefce8"
//                       : "#f0fdf4",
//                 border: `1px solid ${setupStep === 0 ? "#bfdbfe" : setupStep === 1 ? "#fef08a" : "#bbf7d0"}`,
//                 borderRadius: 12,
//                 padding: "14px 16px",
//                 display: "flex",
//                 gap: 10,
//                 alignItems: "flex-start",
//                 transition: "all 0.4s",
//               }}
//             >
//               {setupStep === 0 && (
//                 <Camera
//                   style={{
//                     width: 18,
//                     height: 18,
//                     color: "#2563eb",
//                     flexShrink: 0,
//                     marginTop: 1,
//                   }}
//                 />
//               )}
//               {setupStep === 1 && (
//                 <Crosshair
//                   style={{
//                     width: 18,
//                     height: 18,
//                     color: "#d97706",
//                     flexShrink: 0,
//                     marginTop: 1,
//                   }}
//                 />
//               )}
//               {setupStep === 2 && (
//                 <CheckCircle
//                   style={{
//                     width: 18,
//                     height: 18,
//                     color: "#16a34a",
//                     flexShrink: 0,
//                     marginTop: 1,
//                   }}
//                 />
//               )}
//               <div>
//                 <p
//                   style={{
//                     fontSize: 13,
//                     fontWeight: 600,
//                     margin: "0 0 3px",
//                     color:
//                       setupStep === 0
//                         ? "#1d4ed8"
//                         : setupStep === 1
//                           ? "#b45309"
//                           : "#15803d",
//                   }}
//                 >
//                   {setupStep === 0 && "Step 1: Register your face"}
//                   {setupStep === 1 && "Step 2: Calibrate your gaze"}
//                   {setupStep === 2 && "All done! Ready to start."}
//                 </p>
//                 <p
//                   style={{
//                     fontSize: 12,
//                     color: "#6b7280",
//                     margin: 0,
//                     lineHeight: 1.5,
//                   }}
//                 >
//                   {setupStep === 0 &&
//                     "Look at the camera, center your face in the oval shown in the left panel, then click Capture Face."}
//                   {setupStep === 1 &&
//                     "Click Begin Calibration in the left panel, then follow the moving dot with your eyes until it disappears."}
//                   {setupStep === 2 &&
//                     "Face registered and gaze calibrated. Click Begin Exam when you're ready."}
//                 </p>
//               </div>
//             </div>

//             <button
//               onClick={() => {
//                 setQuizStarted(true);
//                 setScreen("quiz");
//               }}
//               disabled={!calibrationReady}
//               style={{
//                 width: "100%",
//                 padding: "13px 0",
//                 borderRadius: 12,
//                 border: "none",
//                 fontWeight: 700,
//                 fontSize: 14,
//                 cursor: calibrationReady ? "pointer" : "not-allowed",
//                 background: calibrationReady ? "#2563eb" : "#e5e7eb",
//                 color: calibrationReady ? "#fff" : "#9ca3af",
//                 display: "flex",
//                 alignItems: "center",
//                 justifyContent: "center",
//                 gap: 8,
//                 transition: "all 0.3s",
//               }}
//             >
//               <Play style={{ width: 16, height: 16 }} />
//               {calibrationReady ? "Begin Exam" : "Complete setup to continue"}
//             </button>

//             <button
//               onClick={() => {
//                 setCalibrationReady(true);
//                 setSetupStep(2);
//                 setQuizStarted(true);
//                 setScreen("quiz");
//               }}
//               style={{
//                 background: "none",
//                 border: "none",
//                 fontSize: 12,
//                 color: "#9ca3af",
//                 cursor: "pointer",
//                 textDecoration: "underline",
//                 textAlign: "center",
//               }}
//             >
//               Skip camera setup (demo mode)
//             </button>
//           </div>
//         </div>

//         <style>{`@keyframes spin { to { transform: rotate(360deg); } }`}</style>
//       </div>
//     );
//   }

//   // ── Active quiz screen ─────────────────────────────────────────────────────
//   const currentQ = questions[currentIdx];
//   const answeredCount = Object.keys(answers).length;
//   const isLast = currentIdx === questions.length - 1;
//   const urgent = timeLeft < totalTime * 0.25;

//   return (
//     <>
//       {/*
//         The persistent iframe is rendered here at the top level so React never
//         unmounts it when switching screens. In quiz mode it pins itself (via RAF)
//         over the #proctoring-iframe-slot placeholder div in ProctoringPanel.
//       */}
//       {sessionId && (
//         <PersistentProctoringIframe sessionId={sessionId} screen="quiz" />
//       )}

//       <div style={{ minHeight: "100vh", background: "#f9fafb" }}>
//         <div
//           style={{
//             maxWidth: 1100,
//             margin: "0 auto",
//             padding: "32px 20px",
//             display: "flex",
//             gap: 24,
//             alignItems: "flex-start",
//           }}
//         >
//           {/* Left: quiz */}
//           <div style={{ flex: 1, minWidth: 0 }}>
//             <button
//               onClick={handleBack}
//               className="inline-flex items-center gap-2 text-gray-600 hover:text-blue-600 transition mb-6"
//             >
//               <ArrowLeft className="w-4 h-4" /> Back to Course
//             </button>

//             {isFrozen && (
//               <div
//                 style={{
//                   background: "#fef2f2",
//                   border: "1px solid #fca5a5",
//                   borderRadius: 12,
//                   padding: "12px 16px",
//                   marginBottom: 16,
//                   display: "flex",
//                   alignItems: "center",
//                   gap: 10,
//                 }}
//               >
//                 <EyeOff className="w-5 h-5 text-red-500 flex-shrink-0" />
//                 <div>
//                   <p
//                     style={{
//                       fontSize: 14,
//                       fontWeight: 700,
//                       color: "#dc2626",
//                       margin: 0,
//                     }}
//                   >
//                     Exam Paused — Look at the Camera
//                   </p>
//                   <p style={{ fontSize: 12, color: "#b91c1c", margin: 0 }}>
//                     Timer is frozen. Resume by facing the screen.
//                   </p>
//                 </div>
//               </div>
//             )}

//             <div
//               className="bg-white rounded-2xl border border-gray-100 shadow-sm p-6 mb-6"
//               style={{
//                 opacity: isFrozen ? 0.5 : 1,
//                 pointerEvents: isFrozen ? "none" : "auto",
//                 transition: "opacity 0.3s",
//                 userSelect: isFrozen ? "none" : "auto",
//               }}
//             >
//               <div className="flex items-center justify-between mb-4">
//                 <div>
//                   <p className="text-sm text-gray-500 mb-1">{title}</p>
//                   <h1 className="text-xl font-bold text-gray-900">
//                     Question {currentIdx + 1} of {questions.length}
//                   </h1>
//                 </div>
//                 <div
//                   className={`flex items-center gap-2 text-lg font-bold ${urgent ? "text-red-600" : "text-gray-900"}`}
//                 >
//                   <Clock
//                     className={`w-5 h-5 ${urgent ? "animate-pulse text-red-600" : "text-blue-600"}`}
//                   />
//                   {isFrozen ? (
//                     <span style={{ fontSize: 13, color: "#6b7280" }}>
//                       Paused
//                     </span>
//                   ) : (
//                     formatTime(timeLeft)
//                   )}
//                 </div>
//               </div>
//               <TimerBar timeLeft={timeLeft} totalTime={totalTime} />
//               <div className="flex gap-2 mt-4 flex-wrap">
//                 {questions.map((q, i) => (
//                   <button
//                     key={q.id}
//                     onClick={() => setCurrentIdx(i)}
//                     className={`w-8 h-8 rounded-lg text-xs font-semibold transition ${
//                       i === currentIdx
//                         ? "bg-blue-600 text-white"
//                         : answers[q.id] !== undefined
//                           ? "bg-green-100 text-green-700 border border-green-300"
//                           : "bg-gray-100 text-gray-600 hover:bg-gray-200"
//                     }`}
//                   >
//                     {i + 1}
//                   </button>
//                 ))}
//               </div>
//             </div>

//             <div
//               className="bg-white rounded-2xl border border-gray-100 shadow-sm p-6 mb-6"
//               style={{
//                 opacity: isFrozen ? 0.4 : 1,
//                 pointerEvents: isFrozen ? "none" : "auto",
//                 transition: "opacity 0.3s, filter 0.3s",
//                 filter: isFrozen ? "blur(2px)" : "none",
//                 userSelect: isFrozen ? "none" : "auto",
//               }}
//             >
//               <p className="text-lg font-semibold text-gray-900 mb-6">
//                 {currentQ.question_text}
//               </p>
//               <div className="space-y-3">
//                 {currentQ.options.map((opt, i) => {
//                   const selected = answers[currentQ.id] === opt.id;
//                   const letter = String.fromCharCode(65 + i);
//                   return (
//                     <button
//                       key={opt.id}
//                       onClick={() =>
//                         setAnswers((prev) => ({
//                           ...prev,
//                           [currentQ.id]: opt.id,
//                         }))
//                       }
//                       className={`w-full text-left flex items-center gap-4 p-4 rounded-xl border-2 transition-all ${
//                         selected
//                           ? "border-blue-600 bg-blue-50"
//                           : "border-gray-200 bg-white hover:border-blue-300 hover:bg-blue-50/30"
//                       }`}
//                     >
//                       <span
//                         className={`w-8 h-8 flex-shrink-0 flex items-center justify-center rounded-lg font-bold text-sm ${
//                           selected
//                             ? "bg-blue-600 text-white"
//                             : "bg-gray-100 text-gray-600"
//                         }`}
//                       >
//                         {letter}
//                       </span>
//                       <span className="text-gray-900 font-medium">
//                         {opt.option_text}
//                       </span>
//                     </button>
//                   );
//                 })}
//               </div>
//             </div>

//             <div
//               className="flex items-center gap-4 mb-6"
//               style={{
//                 opacity: isFrozen ? 0.4 : 1,
//                 pointerEvents: isFrozen ? "none" : "auto",
//                 transition: "opacity 0.3s",
//               }}
//             >
//               <button
//                 onClick={() => setCurrentIdx((i) => Math.max(0, i - 1))}
//                 disabled={currentIdx === 0}
//                 className="px-4 py-2.5 border border-gray-200 text-gray-700 rounded-xl font-medium hover:bg-gray-50 disabled:opacity-50 disabled:cursor-not-allowed transition flex items-center gap-2"
//               >
//                 <ChevronLeft className="w-4 h-4" /> Prev
//               </button>
//               <div className="flex-1 text-center text-sm text-gray-600 font-medium">
//                 {answeredCount}/{questions.length} answered
//               </div>
//               {isLast ? (
//                 <button
//                   onClick={handleSubmit}
//                   className="px-5 py-2.5 bg-blue-600 text-white rounded-xl font-medium hover:bg-blue-700 transition flex items-center justify-center gap-2"
//                 >
//                   <CheckCircle className="w-4 h-4" /> Submit Quiz
//                 </button>
//               ) : (
//                 <button
//                   onClick={() =>
//                     setCurrentIdx((i) =>
//                       Math.min(questions.length - 1, i + 1),
//                     )
//                   }
//                   className="px-4 py-2.5 border border-gray-200 text-gray-700 rounded-xl font-medium hover:bg-gray-50 transition flex items-center gap-2"
//                 >
//                   Next <ChevronRight className="w-4 h-4" />
//                 </button>
//               )}
//             </div>

//             {answeredCount === questions.length && !isLast && !isFrozen && (
//               <div className="text-center">
//                 <button
//                   onClick={handleSubmit}
//                   className="text-sm font-medium text-blue-600 hover:text-blue-700 underline transition"
//                 >
//                   All answered — submit early?
//                 </button>
//               </div>
//             )}
//           </div>

//           {/* Right: proctoring sidebar (no iframe, just status + log) */}
//           <ProctoringPanel
//             isFrozen={isFrozen}
//             alerts={alerts}
//             attentionState={attentionState}
//             sessionReady={sessionReady}
//           />
//         </div>
//       </div>
//     </>
//   );
// }
