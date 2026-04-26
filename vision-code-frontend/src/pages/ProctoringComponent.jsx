import React, { useEffect, useState, useRef } from "react";

const BASE_URL = "https://examproctoringapi-production.up.railway.app";

const ProctoringComponent = () => {
  const [sessionId, setSessionId] = useState(null);
  const [embedUrl, setEmbedUrl] = useState(null);
  const [alerts, setAlerts] = useState([]);
  const [report, setReport] = useState(null);
  const [warning, setWarning] = useState("");

  // 🔥 track inattentive timing
  const inattentiveStartRef = useRef(null);
  const hasEndedRef = useRef(false);

  // ✅ CREATE SESSION
  const createSession = async () => {
    try {
      const res = await fetch(`${BASE_URL}/api/sessions`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          candidate_id: "cand_001",
          candidate_name: "Test User",
          exam_id: "exam_001",
          exam_label: "Final Exam",
        }),
      });

      const data = await res.json();

      const fixedUrl = `${BASE_URL}/?session_id=${data.session_id}`;

      setSessionId(data.session_id);
      setEmbedUrl(fixedUrl);

      console.log("✅ Session created:", data);
    } catch (err) {
      console.error("❌ Session error:", err);
    }
  };

  // 🔥 END SESSION (safe call once)
  const endSession = async () => {
    if (!sessionId || hasEndedRef.current) return;

    hasEndedRef.current = true;

    try {
      await fetch(`${BASE_URL}/api/sessions/${sessionId}/end`, {
        method: "POST",
      });

      const res = await fetch(`${BASE_URL}/api/sessions/${sessionId}/report`);

      const data = await res.json();
      setReport(data);

      console.log("📊 Final Report:", data);
    } catch (err) {
      console.error("❌ End error:", err);
    }
  };

  // 🔥 REAL-TIME LISTENER (FIXED)
  useEffect(() => {
    const handler = (event) => {
      if (event.origin !== BASE_URL) return;

      const data = event.data;
      if (!data) return;

      // 🚨 Alerts (UI only)
      if (data.type === "alert") {
        setAlerts((prev) => [...prev, data]);
      }

      // 🔥 KEY FIX: Use metrics instead of alerts
      if (data.type === "metrics") {
        const attention = data.state?.attention;

        // ❌ INATTENTIVE
        if (attention === "INATTENTIVE") {
          if (!inattentiveStartRef.current) {
            inattentiveStartRef.current = Date.now();
            setWarning("⚠️ Stay attentive! Exam will end in 5 seconds.");
          } else {
            const duration = Date.now() - inattentiveStartRef.current;

            if (duration >= 5000) {
              console.log("❌ Inattentive > 5s → Ending exam");

              alert("You were inattentive for more than 5 seconds.");

              endSession();

              inattentiveStartRef.current = null;
            }
          }
        }

        // ✅ ATTENTIVE → reset
        if (attention === "ATTENTIVE") {
          inattentiveStartRef.current = null;
          setWarning("");
        }
      }

      // 📊 Session ended from backend
      if (data.type === "session_ended") {
        setReport(data.report);
      }
    };

    window.addEventListener("message", handler);

    return () => {
      window.removeEventListener("message", handler);
    };
  }, [sessionId]);

  // 🔥 INIT
  useEffect(() => {
    createSession();
  }, []);

  return (
    <div style={{ padding: "20px" }}>
      <h2>🎥 AI Proctoring System</h2>

      {/* Warning */}
      {warning && (
        <div style={{ color: "red", marginBottom: "10px" }}>{warning}</div>
      )}

      {/* iframe */}
      {!embedUrl && <p>Creating session...</p>}

      {embedUrl && (
        <iframe
          src={embedUrl}
          title="Proctoring"
          allow="camera; microphone"
          width="100%"
          height="500"
          style={{ border: "2px solid #ccc", borderRadius: "8px" }}
        />
      )}

      {/* End Button */}
      <button onClick={endSession} style={{ marginTop: "10px" }}>
        End Exam
      </button>

      {/* Alerts */}
      <div style={{ marginTop: "20px" }}>
        <h3>🚨 Alerts</h3>
        {alerts.length === 0 && <p>No alerts yet</p>}
        {alerts.map((a, i) => (
          <div key={i}>
            <b>{a.severity}</b> — {a.event} — {a.message}
          </div>
        ))}
      </div>

      {/* Report */}
      {report && (
        <div style={{ marginTop: "20px" }}>
          <h3>📊 Final Result</h3>
          <pre>{JSON.stringify(report.integrity_verdict, null, 2)}</pre>
        </div>
      )}
    </div>
  );
};

export default ProctoringComponent;
