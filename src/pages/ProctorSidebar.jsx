import React, { useEffect, useRef, useState } from "react";

const BASE_URL = "https://examproctoringapi-production.up.railway.app";

export default function ProctorSidebar({ onViolation }) {
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const wsRef = useRef(null);

  const [status, setStatus] = useState("Initializing...");
  const [sessionId, setSessionId] = useState(null);

  const inattentiveStart = useRef(null);

  // ✅ Create session + start camera
  useEffect(() => {
    startSession();
    startCamera();
  }, []);

  const startSession = async () => {
    const res = await fetch(`${BASE_URL}/api/sessions`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        candidate_id: "quiz_user",
        exam_id: "quiz_exam",
      }),
    });

    const data = await res.json();
    setSessionId(data.session_id);

    // 🔥 connect websocket
    const ws = new WebSocket(
      `wss://examproctoringapi-production.up.railway.app/ws/${data.session_id}`,
    );

    wsRef.current = ws;

    ws.onmessage = (msg) => {
      const data = JSON.parse(msg.data);

      if (data.type === "metrics") {
        const attention = data.state?.attention;

        setStatus(attention);

        // 🔥 Inattentive logic
        if (attention === "INATTENTIVE") {
          if (!inattentiveStart.current) {
            inattentiveStart.current = Date.now();
          } else {
            const duration = Date.now() - inattentiveStart.current;

            if (duration > 5000) {
              onViolation(); // 🔥 trigger quiz end
            }
          }
        } else {
          inattentiveStart.current = null;
        }
      }
    };
  };

  // ✅ Start webcam
  const startCamera = async () => {
    const stream = await navigator.mediaDevices.getUserMedia({
      video: true,
    });

    videoRef.current.srcObject = stream;
    videoRef.current.play();

    startSendingFrames();
  };

  // ✅ Send frames continuously
  const startSendingFrames = () => {
    const sendFrame = () => {
      if (!wsRef.current || wsRef.current.readyState !== 1) {
        requestAnimationFrame(sendFrame);
        return;
      }

      const canvas = canvasRef.current;
      const video = videoRef.current;

      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;

      const ctx = canvas.getContext("2d");
      ctx.drawImage(video, 0, 0);

      const base64 = canvas.toDataURL("image/jpeg");

      wsRef.current.send(
        JSON.stringify({
          type: "frame",
          data: base64,
          audio_rms: 0,
        }),
      );

      setTimeout(sendFrame, 100); // ~10 fps
    };

    sendFrame();
  };

  return (
    <div className="w-64 bg-white border rounded-xl p-3 shadow">
      <h3 className="font-bold mb-2">Proctoring</h3>

      <video ref={videoRef} className="w-full rounded mb-2" />

      <div
        className={`text-center font-semibold ${
          status === "ATTENTIVE" ? "text-green-600" : "text-red-600"
        }`}
      >
        {status}
      </div>

      <canvas ref={canvasRef} className="hidden" />
    </div>
  );
}
