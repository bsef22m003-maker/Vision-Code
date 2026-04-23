// import React, { useState, useRef, useEffect } from "react";
// import {
//   Camera,
//   CheckCircle,
//   XCircle,
//   AlertTriangle,
//   Eye,
//   EyeOff,
//   Volume2,
//   VolumeX,
// } from "lucide-react";

// const API_URL = "http://127.0.0.1:8000";

// export default function ProctoringApp() {
//   const [sessionId, setSessionId] = useState(null);
//   const [stage, setStage] = useState("start"); // start, register, calibrate, monitor
//   const [status, setStatus] = useState("NOT STARTED");
//   const [attentionStatus, setAttentionStatus] = useState("UNKNOWN");
//   const [identityStatus, setIdentityStatus] = useState("NOT REGISTERED");
//   const [soundDetected, setSoundDetected] = useState(false);
//   const [warnings, setWarnings] = useState([]);
//   const [violations, setViolations] = useState(0);
//   const [calibrationProgress, setCalibrationProgress] = useState(0);
//   const [message, setMessage] = useState("Click Start to begin proctoring");

//   const videoRef = useRef(null);
//   const canvasRef = useRef(null);
//   const streamRef = useRef(null);
//   const intervalRef = useRef(null);

//   // Initialize camera
//   const startCamera = async () => {
//     try {
//       const stream = await navigator.mediaDevices.getUserMedia({
//         video: { width: 1280, height: 720 },
//       });
//       if (videoRef.current) {
//         videoRef.current.srcObject = stream;
//         streamRef.current = stream;
//       }
//       return true;
//     } catch (error) {
//       setMessage("❌ Camera access denied");
//       return false;
//     }
//   };

//   // Stop camera
//   const stopCamera = () => {
//     if (streamRef.current) {
//       streamRef.current.getTracks().forEach((track) => track.stop());
//     }
//   };

//   // Capture frame as base64
//   const captureFrame = () => {
//     if (!videoRef.current || !canvasRef.current) return null;

//     const video = videoRef.current;
//     const canvas = canvasRef.current;
//     const ctx = canvas.getContext("2d");

//     canvas.width = video.videoWidth;
//     canvas.height = video.videoHeight;
//     ctx.drawImage(video, 0, 0);

//     return canvas.toDataURL("image/jpeg", 0.85);
//   };

//   // Start session
//   const handleStart = async () => {
//     const cameraStarted = await startCamera();
//     if (!cameraStarted) return;

//     try {
//       const response = await fetch(`${API_URL}/api/sessions/create`, {
//         method: "POST",
//         headers: { "Content-Type": "application/json" },
//         body: JSON.stringify({
//           user_id: "student_" + Date.now(),
//           exam_id: "web_exam",
//           use_arcface: false,
//         }),
//       });

//       const data = await response.json();
//       console.log(data);
//       setSessionId(data.session_id);
//       setStage("register");
//       setMessage("Position your face in the frame and click Register");
//     } catch (error) {
//       setMessage("❌ Failed to connect to server. Is it running?");
//     }
//   };

//   // Register face
//   const handleRegister = async () => {
//     setMessage("Registering face...");
//     const imageData = captureFrame();

//     try {
//       const response = await fetch(
//         `${API_URL}/api/sessions/${sessionId}/register-face-base64`,
//         {
//           method: "POST",
//           headers: { "Content-Type": "application/json" },
//           body: JSON.stringify({ image: imageData }),
//         }
//       );

//       const data = await response.json();

//       if (data.success) {
//         setIdentityStatus("REGISTERED");
//         setStage("calibrate");
//         setMessage(
//           "Look at the CENTER of your screen. Calibration starting..."
//         );
//         setTimeout(() => startCalibration(), 2000);
//       } else {
//         setMessage("❌ " + data.message);
//       }
//     } catch (error) {
//       setMessage("❌ Registration failed");
//     }
//   };

//   // Calibration
//   const startCalibration = async () => {
//     setMessage("Calibrating... Keep looking at the screen!");
//     const calibrationData = { samples: [] };
//     const duration = 5000; // 5 seconds
//     const startTime = new Date();

//     const calibrationInterval = setInterval(() => {
//       const elapsed = new Date() - startTime;
//       setCalibrationProgress(Math.min((elapsed / duration) * 100, 100));

//       if (elapsed >= duration) {
//         clearInterval(calibrationInterval);
//         sendCalibration();
//       }
//     }, 100);
//   };

//   const sendCalibration = async () => {
//     // For simplicity, using default baseline values
//     // In production, you'd collect actual gaze data
//     try {
//       await fetch(`${API_URL}/api/sessions/${sessionId}/calibrate`, {
//         method: "POST",
//         headers: { "Content-Type": "application/json" },
//         body: JSON.stringify({
//           baseline_yaw: 0.0,
//           baseline_pitch: 0.0,
//           baseline_gaze_x: 0.5,
//           baseline_gaze_y: 0.5,
//         }),
//       });

//       setStage("monitor");
//       setMessage("✅ Monitoring started");
//       startMonitoring();
//     } catch (error) {
//       setMessage("❌ Calibration failed");
//     }
//   };

//   // Monitoring
//   const startMonitoring = () => {
//     intervalRef.current = setInterval(async () => {
//       const imageData = captureFrame();
//       if (!imageData) return;

//       try {
//         const response = await fetch(
//           `${API_URL}/api/sessions/${sessionId}/analyze-frame-base64`,
//           {
//             method: "POST",
//             headers: { "Content-Type": "application/json" },
//             body: JSON.stringify({ image: imageData }),
//           }
//         );

//         const data = await response.json();

//         setAttentionStatus(data.attention_status);
//         setIdentityStatus(data.identity_status);
//         setSoundDetected(data.sound_detected);
//         setWarnings(data.warnings || []);

//         if (data.warnings && data.warnings.length > 0) {
//           setViolations((prev) => prev + 1);
//         }
//       } catch (error) {
//         console.error("Analysis failed:", error);
//       }
//     }, 2000); // Analyze every 2 seconds
//   };

//   // Stop monitoring
//   const handleStop = async () => {
//     if (intervalRef.current) {
//       clearInterval(intervalRef.current);
//     }

//     if (sessionId) {
//       try {
//         await fetch(`${API_URL}/api/sessions/${sessionId}`, {
//           method: "DELETE",
//         });
//       } catch (error) {
//         console.error("Failed to end session");
//       }
//     }

//     stopCamera();
//     setStage("start");
//     setSessionId(null);
//     setAttentionStatus("UNKNOWN");
//     setIdentityStatus("NOT REGISTERED");
//     setWarnings([]);
//     setViolations(0);
//     setMessage("Session ended. Click Start to begin again.");
//   };

//   // Cleanup on unmount
//   useEffect(() => {
//     return () => {
//       stopCamera();
//       if (intervalRef.current) {
//         clearInterval(intervalRef.current);
//       }
//     };
//   }, []);

//   return (
//     <div className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-100 p-6">
//       <div className="max-w-6xl mx-auto">
//         {/* Header */}
//         <div className="bg-white rounded-lg shadow-lg p-6 mb-6">
//           <h1 className="text-3xl font-bold text-gray-800 flex items-center gap-3">
//             <Camera className="w-8 h-8 text-indigo-600" />
//             Exam Proctoring System
//           </h1>
//           <p className="text-gray-600 mt-2">{message}</p>
//         </div>

//         <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
//           {/* Video Feed */}
//           <div className="lg:col-span-2">
//             <div className="bg-white rounded-lg shadow-lg p-6">
//               <div className="relative bg-gray-900 rounded-lg overflow-hidden aspect-video">
//                 <video
//                   ref={videoRef}
//                   autoPlay
//                   playsInline
//                   muted
//                   className="w-full h-full object-cover"
//                 />
//                 <canvas ref={canvasRef} className="hidden" />

//                 {/* Overlay for registration */}
//                 {stage === "register" && (
//                   <div className="absolute inset-0 flex items-center justify-center">
//                     <div className="w-64 h-64 border-4 border-yellow-400 rounded-lg"></div>
//                   </div>
//                 )}

//                 {/* Calibration overlay */}
//                 {stage === "calibrate" && (
//                   <div className="absolute inset-0 flex items-center justify-center">
//                     <div className="text-center">
//                       <div className="w-16 h-16 mx-auto mb-4">
//                         <div className="w-full h-1 bg-yellow-400 mb-2"></div>
//                         <div className="w-1 h-full bg-yellow-400 mx-auto"></div>
//                       </div>
//                       <div className="text-yellow-400 text-xl font-bold">
//                         {calibrationProgress.toFixed(0)}%
//                       </div>
//                     </div>
//                   </div>
//                 )}

//                 {/* Status overlay */}
//                 {stage === "monitor" && (
//                   <div className="absolute top-4 left-4 right-4">
//                     <div
//                       className={`px-4 py-2 rounded-lg font-semibold ${
//                         attentionStatus === "ATTENTIVE"
//                           ? "bg-green-500 text-white"
//                           : "bg-red-500 text-white"
//                       }`}
//                     >
//                       {attentionStatus}
//                     </div>
//                   </div>
//                 )}
//               </div>

//               {/* Controls */}
//               <div className="mt-4 flex gap-3">
//                 {stage === "start" && (
//                   <button
//                     onClick={handleStart}
//                     className="flex-1 bg-indigo-600 text-white px-6 py-3 rounded-lg font-semibold hover:bg-indigo-700 transition"
//                   >
//                     Start Proctoring
//                   </button>
//                 )}

//                 {stage === "register" && (
//                   <button
//                     onClick={handleRegister}
//                     className="flex-1 bg-green-600 text-white px-6 py-3 rounded-lg font-semibold hover:bg-green-700 transition"
//                   >
//                     Register Face
//                   </button>
//                 )}

//                 {stage === "monitor" && (
//                   <button
//                     onClick={handleStop}
//                     className="flex-1 bg-red-600 text-white px-6 py-3 rounded-lg font-semibold hover:bg-red-700 transition"
//                   >
//                     Stop Monitoring
//                   </button>
//                 )}
//               </div>
//             </div>
//           </div>

//           {/* Status Panel */}
//           <div className="space-y-6">
//             {/* Attention Status */}
//             <div className="bg-white rounded-lg shadow-lg p-6">
//               <h3 className="text-lg font-semibold mb-4 flex items-center gap-2">
//                 <Eye className="w-5 h-5" />
//                 Attention
//               </h3>
//               <div
//                 className={`flex items-center gap-3 p-4 rounded-lg ${
//                   attentionStatus === "ATTENTIVE"
//                     ? "bg-green-100 text-green-800"
//                     : attentionStatus === "INATTENTIVE"
//                     ? "bg-red-100 text-red-800"
//                     : "bg-gray-100 text-gray-800"
//                 }`}
//               >
//                 {attentionStatus === "ATTENTIVE" ? (
//                   <CheckCircle className="w-6 h-6" />
//                 ) : attentionStatus === "INATTENTIVE" ? (
//                   <XCircle className="w-6 h-6" />
//                 ) : (
//                   <AlertTriangle className="w-6 h-6" />
//                 )}
//                 <span className="font-semibold">{attentionStatus}</span>
//               </div>
//             </div>

//             {/* Identity Status */}
//             <div className="bg-white rounded-lg shadow-lg p-6">
//               <h3 className="text-lg font-semibold mb-4">Identity</h3>
//               <div
//                 className={`flex items-center gap-3 p-4 rounded-lg ${
//                   identityStatus === "VERIFIED"
//                     ? "bg-green-100 text-green-800"
//                     : "bg-gray-100 text-gray-800"
//                 }`}
//               >
//                 {identityStatus === "VERIFIED" ? (
//                   <CheckCircle className="w-6 h-6" />
//                 ) : (
//                   <AlertTriangle className="w-6 h-6" />
//                 )}
//                 <span className="font-semibold">{identityStatus}</span>
//               </div>
//             </div>

//             {/* Sound Status */}
//             <div className="bg-white rounded-lg shadow-lg p-6">
//               <h3 className="text-lg font-semibold mb-4">Audio</h3>
//               <div
//                 className={`flex items-center gap-3 p-4 rounded-lg ${
//                   soundDetected
//                     ? "bg-red-100 text-red-800"
//                     : "bg-green-100 text-green-800"
//                 }`}
//               >
//                 {soundDetected ? (
//                   <Volume2 className="w-6 h-6" />
//                 ) : (
//                   <VolumeX className="w-6 h-6" />
//                 )}
//                 <span className="font-semibold">
//                   {soundDetected ? "Sound Detected" : "Quiet"}
//                 </span>
//               </div>
//             </div>

//             {/* Violations */}
//             <div className="bg-white rounded-lg shadow-lg p-6">
//               <h3 className="text-lg font-semibold mb-4">Violations</h3>
//               <div className="text-3xl font-bold text-red-600 text-center">
//                 {violations}
//               </div>

//               {warnings.length > 0 && (
//                 <div className="mt-4 space-y-2">
//                   {warnings.map((warning, idx) => (
//                     <div
//                       key={idx}
//                       className="bg-red-50 border border-red-200 rounded p-3"
//                     >
//                       <p className="text-sm text-red-800">⚠ {warning}</p>
//                     </div>
//                   ))}
//                 </div>
//               )}
//             </div>
//           </div>
//         </div>

//         {/* Info */}
//         <div className="mt-6 bg-blue-50 border border-blue-200 rounded-lg p-4">
//           <h4 className="font-semibold text-blue-900 mb-2">How it works:</h4>
//           <ol className="text-sm text-blue-800 space-y-1 list-decimal list-inside">
//             <li>Click "Start Proctoring" to begin</li>
//             <li>Register your face by clicking "Register Face"</li>
//             <li>Look at the screen center during calibration</li>
//             <li>Monitoring will start automatically</li>
//             <li>Stay focused on the screen to avoid violations</li>
//           </ol>
//         </div>
//       </div>
//     </div>
//   );
// }
