"""
FastAPI Server for Exam Proctoring System
Provides RESTful API endpoints for proctoring functionality
"""
from fastapi import FastAPI, File, UploadFile, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel
from typing import Optional, Dict, List
import cv2
import numpy as np
import base64
import io
from datetime import datetime
import asyncio
import uuid
import json

# Import your existing modules
from gaze_tracker import GazeTracker
from attention_analyzer import AttentionAnalyzer
from face_recognizer import OpenCVFaceRecognizer, ArcFaceRecognizer
from sound_detector import SoundDetector
from metrics import PerformanceMetrics
from data_logger import DataLogger
from config import *

# Initialize FastAPI app
app = FastAPI(
    title="Exam Proctoring API",
    description="Real-time exam proctoring with face recognition, gaze tracking, and attention monitoring",
    version="1.0.0"
)

# Enable CORS for web clients
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Store active sessions
active_sessions: Dict[str, dict] = {}


# ==================== Pydantic Models ====================

class SessionCreate(BaseModel):
    user_id: str
    exam_id: str
    use_arcface: bool = False
    sound_threshold: float = SOUND_THRESHOLD


class SessionResponse(BaseModel):
    session_id: str
    user_id: str
    exam_id: str
    created_at: str
    status: str


class CalibrationData(BaseModel):
    baseline_yaw: float
    baseline_pitch: float
    baseline_gaze_x: float
    baseline_gaze_y: float


class FrameAnalysisResponse(BaseModel):
    session_id: str
    timestamp: str
    attention_status: str
    attention_color: List[int]
    candidate_status: str
    identity_status: str
    identity_verified: bool
    sound_detected: bool
    sound_level: float
    tracking_details: Optional[dict]
    warnings: List[str]


class MetricsResponse(BaseModel):
    total_frames: int
    attentive_frames: int
    inattentive_frames: int
    attention_rate: float
    precision: float
    recall: float
    f1_score: float
    accuracy: float
    sound_detections: int
    face_verifications_passed: int
    face_verifications_failed: int


# ==================== Helper Functions ====================

def decode_base64_image(base64_string: str) -> np.ndarray:
    """Decode base64 image to numpy array"""
    try:
        # Remove data URL prefix if present
        if ',' in base64_string:
            base64_string = base64_string.split(',')[1]
        
        img_data = base64.b64decode(base64_string)
        img_array = np.frombuffer(img_data, dtype=np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        return img
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image data: {str(e)}")


def encode_image_to_base64(img: np.ndarray) -> str:
    """Encode numpy array to base64 string"""
    _, buffer = cv2.imencode('.jpg', img)
    img_base64 = base64.b64encode(buffer).decode('utf-8')
    return f"data:image/jpeg;base64,{img_base64}"


def initialize_session_components(session_id: str, use_arcface: bool = False, sound_threshold: float = SOUND_THRESHOLD):
    """Initialize all components for a session"""
    try:
        # Initialize components
        gaze_tracker = GazeTracker()
        
        if use_arcface:
            face_recognizer = ArcFaceRecognizer()
        else:
            face_recognizer = OpenCVFaceRecognizer()
        
        sound_detector = SoundDetector(threshold=sound_threshold)
        sound_detector.start()
        
        attention_analyzer = AttentionAnalyzer()
        metrics = PerformanceMetrics()
        
        csv_filename = f"session_{session_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        data_logger = DataLogger(csv_filename)
        
        return {
            'gaze_tracker': gaze_tracker,
            'face_recognizer': face_recognizer,
            'sound_detector': sound_detector,
            'attention_analyzer': attention_analyzer,
            'metrics': metrics,
            'data_logger': data_logger,
            'csv_filename': csv_filename,
            'calibrated': False,
            'face_registered': False
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to initialize session: {str(e)}")


# ==================== API Endpoints ====================

@app.get("/")
async def root():
    """API health check"""
    return {
        "status": "online",
        "service": "Exam Proctoring API",
        "version": "1.0.0",
        "active_sessions": len(active_sessions)
    }


@app.post("/api/sessions/create", response_model=SessionResponse)
async def create_session(session_data: SessionCreate):
    """Create a new proctoring session"""
    session_id = str(uuid.uuid4())
    
    try:
        components = initialize_session_components(
            session_id, 
            session_data.use_arcface,
            session_data.sound_threshold
        )
        
        active_sessions[session_id] = {
            'session_id': session_id,
            'user_id': session_data.user_id,
            'exam_id': session_data.exam_id,
            'created_at': datetime.now().isoformat(),
            'status': 'created',
            'components': components
        }
        
        return SessionResponse(
            session_id=session_id,
            user_id=session_data.user_id,
            exam_id=session_data.exam_id,
            created_at=active_sessions[session_id]['created_at'],
            status='created'
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/sessions/{session_id}/register-face")
async def register_face(session_id: str, file: UploadFile = File(...)):
    """Register user's face for identity verification"""
    if session_id not in active_sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    try:
        # Read and decode image
        contents = await file.read()
        img_array = np.frombuffer(contents, dtype=np.uint8)
        frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        
        if frame is None:
            raise HTTPException(status_code=400, detail="Invalid image file")
        
        # Register face
        face_recognizer = active_sessions[session_id]['components']['face_recognizer']
        success, message = face_recognizer.register_face(frame)
        
        if success:
            active_sessions[session_id]['components']['face_registered'] = True
            active_sessions[session_id]['status'] = 'face_registered'
        
        return {
            "success": success,
            "message": message,
            "session_id": session_id
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/sessions/{session_id}/register-face-base64")
async def register_face_base64(session_id: str, image_data: dict):
    """Register user's face using base64 encoded image"""
    if session_id not in active_sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    try:
        frame = decode_base64_image(image_data.get('image', ''))
        
        face_recognizer = active_sessions[session_id]['components']['face_recognizer']
        success, message = face_recognizer.register_face(frame)
        
        if success:
            active_sessions[session_id]['components']['face_registered'] = True
            active_sessions[session_id]['status'] = 'face_registered'
        
        return {
            "success": success,
            "message": message,
            "session_id": session_id
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/sessions/{session_id}/calibrate")
async def calibrate_session(session_id: str, calibration: CalibrationData):
    """Calibrate attention baseline for the session"""
    if session_id not in active_sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    try:
        attention_analyzer = active_sessions[session_id]['components']['attention_analyzer']
        attention_analyzer.update_baseline(
            calibration.baseline_yaw,
            calibration.baseline_pitch,
            calibration.baseline_gaze_x,
            calibration.baseline_gaze_y
        )
        
        active_sessions[session_id]['components']['calibrated'] = True
        active_sessions[session_id]['status'] = 'calibrated'
        
        return {
            "success": True,
            "message": "Calibration successful",
            "baseline": {
                "yaw": calibration.baseline_yaw,
                "pitch": calibration.baseline_pitch,
                "gaze_x": calibration.baseline_gaze_x,
                "gaze_y": calibration.baseline_gaze_y
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/sessions/{session_id}/analyze-frame", response_model=FrameAnalysisResponse)
async def analyze_frame(session_id: str, file: UploadFile = File(...)):
    """Analyze a single frame for proctoring violations"""
    if session_id not in active_sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    session = active_sessions[session_id]
    components = session['components']
    
    if not components['face_registered']:
        raise HTTPException(status_code=400, detail="Face not registered. Call /register-face first")
    
    if not components['calibrated']:
        raise HTTPException(status_code=400, detail="Session not calibrated. Call /calibrate first")
    
    try:
        # Read and decode image
        contents = await file.read()
        img_array = np.frombuffer(contents, dtype=np.uint8)
        frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        
        if frame is None:
            raise HTTPException(status_code=400, detail="Invalid image file")
        
        # Process frame
        tracking_data = components['gaze_tracker'].process_frame(frame)
        attention_result = components['attention_analyzer'].analyze_frame(tracking_data)
        
        # Verify identity (periodic check)
        identity_status, identity_color, match_conf = components['face_recognizer'].get_status()
        verify_result, verify_msg = components['face_recognizer'].verify_face(frame)
        
        if verify_result is not None:
            components['metrics'].update_identity(verify_result)
        
        # Get sound status
        sound_details = components['sound_detector'].get_detailed_status()
        components['metrics'].update_sound(sound_details['is_detected'])
        
        # Update metrics
        predicted_attentive = attention_result['status'] == "ATTENTIVE"
        actual_attentive = attention_result['candidate_status'].startswith("ATTENTIVE")
        components['metrics'].update(predicted_attentive, actual_attentive)
        
        # Log data
        components['data_logger'].log_data({
            'timestamp': datetime.now().isoformat(),
            'session_time': (datetime.now() - datetime.fromisoformat(session['created_at'])).total_seconds(),
            'attention_state': attention_result['status'],
            'identity_status': identity_status,
            'sound_detected': sound_details['is_detected'],
            'sound_level': sound_details['rms_level'],
            'head_yaw': attention_result['details']['head_yaw'] if attention_result['details'] else 0.0,
            'head_pitch': attention_result['details']['head_pitch'] if attention_result['details'] else 0.0,
            'gaze_x': tracking_data['gaze_x'] if tracking_data else 0.0,
            'gaze_y': tracking_data['gaze_y'] if tracking_data else 0.0,
            'ear': tracking_data['ear'] if tracking_data else 0.0,
            'confidence': match_conf
        })
        
        # Determine warnings
        warnings = []
        if attention_result['status'] == "INATTENTIVE":
            warnings.append("Student not paying attention")
        if identity_status == "DIFFERENT PERSON":
            warnings.append("Identity verification failed - different person detected")
        elif identity_status == "MULTIPLE FACES":
            warnings.append("Multiple faces detected")
        elif identity_status == "NO FACE":
            warnings.append("No face detected")
        if sound_details['is_detected']:
            warnings.append("Sound/speech detected")
        
        return FrameAnalysisResponse(
            session_id=session_id,
            timestamp=datetime.now().isoformat(),
            attention_status=attention_result['status'],
            attention_color=list(attention_result['color']),
            candidate_status=attention_result['candidate_status'],
            identity_status=identity_status,
            identity_verified=(identity_status == "VERIFIED"),
            sound_detected=sound_details['is_detected'],
            sound_level=sound_details['rms_level'],
            tracking_details=attention_result['details'],
            warnings=warnings
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/sessions/{session_id}/analyze-frame-base64", response_model=FrameAnalysisResponse)
async def analyze_frame_base64(session_id: str, image_data: dict):
    """Analyze a frame sent as base64 encoded image"""
    if session_id not in active_sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    session = active_sessions[session_id]
    components = session['components']
    
    if not components['face_registered']:
        raise HTTPException(status_code=400, detail="Face not registered")
    
    if not components['calibrated']:
        raise HTTPException(status_code=400, detail="Session not calibrated")
    
    try:
        frame = decode_base64_image(image_data.get('image', ''))
        
        # Process frame (same logic as analyze_frame)
        tracking_data = components['gaze_tracker'].process_frame(frame)
        attention_result = components['attention_analyzer'].analyze_frame(tracking_data)
        
        identity_status, identity_color, match_conf = components['face_recognizer'].get_status()
        verify_result, verify_msg = components['face_recognizer'].verify_face(frame)
        
        if verify_result is not None:
            components['metrics'].update_identity(verify_result)
        
        sound_details = components['sound_detector'].get_detailed_status()
        components['metrics'].update_sound(sound_details['is_detected'])
        
        predicted_attentive = attention_result['status'] == "ATTENTIVE"
        actual_attentive = attention_result['candidate_status'].startswith("ATTENTIVE")
        components['metrics'].update(predicted_attentive, actual_attentive)
        
        components['data_logger'].log_data({
            'timestamp': datetime.now().isoformat(),
            'session_time': (datetime.now() - datetime.fromisoformat(session['created_at'])).total_seconds(),
            'attention_state': attention_result['status'],
            'identity_status': identity_status,
            'sound_detected': sound_details['is_detected'],
            'sound_level': sound_details['rms_level'],
            'head_yaw': attention_result['details']['head_yaw'] if attention_result['details'] else 0.0,
            'head_pitch': attention_result['details']['head_pitch'] if attention_result['details'] else 0.0,
            'gaze_x': tracking_data['gaze_x'] if tracking_data else 0.0,
            'gaze_y': tracking_data['gaze_y'] if tracking_data else 0.0,
            'ear': tracking_data['ear'] if tracking_data else 0.0,
            'confidence': match_conf
        })
        
        warnings = []
        if attention_result['status'] == "INATTENTIVE":
            warnings.append("Student not paying attention")
        if identity_status == "DIFFERENT PERSON":
            warnings.append("Identity verification failed")
        elif identity_status == "MULTIPLE FACES":
            warnings.append("Multiple faces detected")
        elif identity_status == "NO FACE":
            warnings.append("No face detected")
        if sound_details['is_detected']:
            warnings.append("Sound/speech detected")
        
        return FrameAnalysisResponse(
            session_id=session_id,
            timestamp=datetime.now().isoformat(),
            attention_status=attention_result['status'],
            attention_color=list(attention_result['color']),
            candidate_status=attention_result['candidate_status'],
            identity_status=identity_status,
            identity_verified=(identity_status == "VERIFIED"),
            sound_detected=sound_details['is_detected'],
            sound_level=sound_details['rms_level'],
            tracking_details=attention_result['details'],
            warnings=warnings
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/sessions/{session_id}/metrics", response_model=MetricsResponse)
async def get_session_metrics(session_id: str):
    """Get performance metrics for a session"""
    if session_id not in active_sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    try:
        metrics = active_sessions[session_id]['components']['metrics']
        calculated_metrics = metrics.calculate_metrics()
        
        return MetricsResponse(
            total_frames=metrics.total_frames,
            attentive_frames=metrics.attentive_frames,
            inattentive_frames=metrics.inattentive_frames,
            attention_rate=metrics.attentive_frames / metrics.total_frames if metrics.total_frames > 0 else 0.0,
            precision=calculated_metrics['precision'],
            recall=calculated_metrics['recall'],
            f1_score=calculated_metrics['f1_score'],
            accuracy=calculated_metrics['accuracy'],
            sound_detections=metrics.sound_detected_count,
            face_verifications_passed=metrics.face_verified_count,
            face_verifications_failed=metrics.face_failed_count
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/sessions/{session_id}/status")
async def get_session_status(session_id: str):
    """Get current status of a session"""
    if session_id not in active_sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    session = active_sessions[session_id]
    components = session['components']
    
    return {
        "session_id": session_id,
        "user_id": session['user_id'],
        "exam_id": session['exam_id'],
        "status": session['status'],
        "created_at": session['created_at'],
        "face_registered": components['face_registered'],
        "calibrated": components['calibrated'],
        "csv_filename": components['csv_filename']
    }


@app.delete("/api/sessions/{session_id}")
async def end_session(session_id: str):
    """End a proctoring session and cleanup resources"""
    if session_id not in active_sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    try:
        components = active_sessions[session_id]['components']
        
        # Cleanup resources
        components['sound_detector'].stop()
        components['data_logger'].finalize()
        components['gaze_tracker'].close()
        
        # Get final metrics
        metrics = components['metrics']
        final_report = metrics.get_summary_report()
        
        # Remove session
        csv_filename = components['csv_filename']
        del active_sessions[session_id]
        
        return {
            "success": True,
            "message": "Session ended successfully",
            "session_id": session_id,
            "csv_filename": csv_filename,
            "final_metrics": {
                "total_frames": metrics.total_frames,
                "attention_rate": metrics.attentive_frames / metrics.total_frames if metrics.total_frames > 0 else 0.0
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/sessions")
async def list_sessions():
    """List all active sessions"""
    return {
        "active_sessions": len(active_sessions),
        "sessions": [
            {
                "session_id": sid,
                "user_id": session['user_id'],
                "exam_id": session['exam_id'],
                "status": session['status'],
                "created_at": session['created_at']
            }
            for sid, session in active_sessions.items()
        ]
    }


@app.post("/api/sessions/{session_id}/log-data")
async def log_data(session_id: str, log_data: dict):
    """Simple endpoint to log data from local detection"""
    if session_id not in active_sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    try:
        components = active_sessions[session_id]['components']
        components['data_logger'].log_data(log_data)
        return {"success": True}
    except Exception as e:
        return {"success": False, "error": str(e)}


@app.websocket("/ws/sessions/{session_id}")
async def websocket_endpoint(websocket: WebSocket, session_id: str):
    """WebSocket endpoint for real-time proctoring"""
    await websocket.accept()
    
    if session_id not in active_sessions:
        await websocket.send_json({"error": "Session not found"})
        await websocket.close()
        return
    
    session = active_sessions[session_id]
    components = session['components']
    
    try:
        while True:
            # Receive frame data
            data = await websocket.receive_json()
            
            if data.get('type') == 'frame':
                try:
                    frame = decode_base64_image(data.get('image', ''))
                    
                    # Process frame
                    tracking_data = components['gaze_tracker'].process_frame(frame)
                    attention_result = components['attention_analyzer'].analyze_frame(tracking_data)
                    
                    identity_status, identity_color, match_conf = components['face_recognizer'].get_status()
                    sound_details = components['sound_detector'].get_detailed_status()
                    
                    # Send response
                    await websocket.send_json({
                        "type": "analysis",
                        "timestamp": datetime.now().isoformat(),
                        "attention_status": attention_result['status'],
                        "candidate_status": attention_result['candidate_status'],
                        "identity_status": identity_status,
                        "sound_detected": sound_details['is_detected'],
                        "sound_level": sound_details['rms_level']
                    })
                    
                except Exception as e:
                    await websocket.send_json({"error": str(e)})
            
            elif data.get('type') == 'ping':
                await websocket.send_json({"type": "pong"})
                
    except WebSocketDisconnect:
        print(f"WebSocket disconnected for session {session_id}")
    except Exception as e:
        print(f"WebSocket error: {e}")
        await websocket.close()


# Run with: uvicorn api_server:app --reload --host 0.0.0.0 --port 8000
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)