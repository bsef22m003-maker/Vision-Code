"""
Real-time Monitoring with API Session Management
Runs detection LOCALLY (every frame) for instant response
Uses API only for session management and data logging
"""

import cv2
import time
import requests
import numpy as np
from datetime import datetime
from threading import Thread, Lock
import queue

# Import your local modules (runs on YOUR computer, not via API)
from gaze_tracker import GazeTracker
from attention_analyzer import AttentionAnalyzer
from sound_detector import SoundDetector
from face_recognizer import OpenCVFaceRecognizer
from config import *

API_URL = "http://localhost:8000"

# Queue for async API logging (doesn't slow down detection)
log_queue = queue.Queue(maxsize=100)
api_logger_running = True

def api_logging_worker(session_id):
    """Background thread that logs to API without blocking detection"""
    global api_logger_running
    
    while api_logger_running:
        try:
            log_data = log_queue.get(timeout=1)
            if log_data is None:  # Stop signal
                break
            
            # Send to API (non-blocking, don't care if it fails)
            try:
                requests.post(
                    f"{API_URL}/api/sessions/{session_id}/log-data",
                    json=log_data,
                    timeout=1
                )
            except:
                pass  # Don't let API failures affect monitoring
                
        except queue.Empty:
            continue

def register_face_local(cap, face_recognizer):
    """Register face locally (instant, no API delay)"""
    print("\n" + "="*60)
    print("📸 FACE REGISTRATION")
    print("="*60)
    print("Press SPACE to capture your face")
    print("="*60 + "\n")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            continue
        
        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]
        
        # Guide box
        box_size = min(w, h) // 2
        box_x = (w - box_size) // 2
        box_y = (h - box_size) // 2
        
        cv2.rectangle(frame, (box_x, box_y), (box_x + box_size, box_y + box_size), 
                     (0, 255, 255), 3)
        cv2.putText(frame, "Position face in box - Press SPACE", (w//2-250, 50),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        cv2.imshow("Face Registration", frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == 32:  # SPACE
            success, message = face_recognizer.register_face(frame)
            print(f"{'✅' if success else '❌'} {message}")
            if success:
                cv2.destroyWindow("Face Registration")
                return True
            time.sleep(1)
        elif key == 27:  # ESC
            return False
    
    return False

def calibrate_local(cap, gaze_tracker, duration=3.0):
    """Calibrate locally (instant detection)"""
    print("\n" + "="*60)
    print("🎯 CALIBRATION")
    print("="*60)
    print(f"Look at the screen CENTER for {duration} seconds")
    print("="*60 + "\n")
    
    yaw_list, pitch_list, gx_list, gy_list = [], [], [], []
    start_time = time.time()
    
    while time.time() - start_time < duration:
        ret, frame = cap.read()
        if not ret:
            continue
        
        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]
        
        tracking_data = gaze_tracker.process_frame(frame)
        
        if tracking_data:
            yaw_list.append(tracking_data['yaw'] or 0.0)
            pitch_list.append(tracking_data['pitch'] or 0.0)
            gx_list.append(tracking_data['gaze_x'])
            gy_list.append(tracking_data['gaze_y'])
        
        elapsed = time.time() - start_time
        remaining = duration - elapsed
        progress = int((elapsed / duration) * 100)
        
        # Crosshair at center
        center_x, center_y = w // 2, h // 2
        cv2.line(frame, (center_x - 30, center_y), (center_x + 30, center_y), 
                (0, 255, 255), 3)
        cv2.line(frame, (center_x, center_y - 30), (center_x, center_y + 30), 
                (0, 255, 255), 3)
        cv2.circle(frame, (center_x, center_y), 50, (0, 255, 255), 2)
        
        cv2.putText(frame, f"Calibrating... {remaining:.1f}s [{progress}%]", 
                   (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        cv2.putText(frame, "Look at the + sign!", (20, 90),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        
        cv2.imshow("Calibration", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            return None, None, None, None
    
    cv2.destroyWindow("Calibration")
    
    baseline_yaw = float(np.median(yaw_list)) if yaw_list else 0.0
    baseline_pitch = float(np.median(pitch_list)) if pitch_list else 0.0
    baseline_gaze_x = float(np.median(gx_list)) if gx_list else 0.5
    baseline_gaze_y = float(np.median(gy_list)) if gy_list else 0.5
    
    print(f"✅ Calibration complete:")
    print(f"   Samples: {len(yaw_list)}")
    print(f"   Yaw={baseline_yaw:.2f}°, Pitch={baseline_pitch:.2f}°")
    print(f"   Gaze=({baseline_gaze_x:.3f}, {baseline_gaze_y:.3f})\n")
    
    return baseline_yaw, baseline_pitch, baseline_gaze_x, baseline_gaze_y

def main():
    global api_logger_running
    
    print("\n" + "="*70)
    print("🎓 REAL-TIME EXAM PROCTORING (Local Detection + API Logging)")
    print("="*70)
    print("Detection runs LOCALLY = Instant response (like original code)")
    print("API used only for session management and logging")
    print("="*70 + "\n")
    
    # Step 1: Create API session (optional - for logging only)
    session_id = None
    try:
        response = requests.post(f"{API_URL}/api/sessions/create", 
                                json={
                                    "user_id": "student123",
                                    "exam_id": "realtime_test",
                                    "use_arcface": False
                                }, timeout=3)
        session_id = response.json()['session_id']
        print(f"✅ API session created: {session_id}")
        
        # Start background logging thread
        log_thread = Thread(target=api_logging_worker, args=(session_id,), daemon=True)
        log_thread.start()
    except:
        print("⚠ API not available - will run without logging")
        session_id = None
    
    # Step 2: Initialize LOCAL components (runs on your computer)
    print("\nInitializing local detection components...")
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Cannot open camera")
        return
    
    # Set high quality for good detection
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 30)
    
    gaze_tracker = GazeTracker()
    face_recognizer = OpenCVFaceRecognizer()
    sound_detector = SoundDetector()
    sound_detector.start()
    
    print("✅ Local components initialized\n")
    
    # Step 3: Register face locally
    if not register_face_local(cap, face_recognizer):
        print("Registration cancelled")
        cap.release()
        cv2.destroyAllWindows()
        return
    
    # Step 4: Calibrate locally
    baseline_yaw, baseline_pitch, baseline_gaze_x, baseline_gaze_y = calibrate_local(cap, gaze_tracker)
    
    if baseline_yaw is None:
        print("Calibration cancelled")
        cap.release()
        cv2.destroyAllWindows()
        gaze_tracker.close()
        sound_detector.stop()
        return
    
    # Initialize attention analyzer with baseline
    attention_analyzer = AttentionAnalyzer(baseline_yaw, baseline_pitch, 
                                          baseline_gaze_x, baseline_gaze_y)
    
    # Send calibration to API (if available)
    if session_id:
        try:
            requests.post(f"{API_URL}/api/sessions/{session_id}/calibrate",
                         json={
                             "baseline_yaw": baseline_yaw,
                             "baseline_pitch": baseline_pitch,
                             "baseline_gaze_x": baseline_gaze_x,
                             "baseline_gaze_y": baseline_gaze_y
                         }, timeout=2)
        except:
            pass
    
    # Step 5: Real-time monitoring
    print("="*70)
    print("🎥 REAL-TIME MONITORING STARTED")
    print("="*70)
    print("Press ESC to stop")
    print("="*70 + "\n")
    
    start_time = time.time()
    frame_count = 0
    violation_count = 0
    last_log_time = start_time
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                continue
            
            frame = cv2.flip(frame, 1)
            frame_count += 1
            current_time = time.time()
            h, w = frame.shape[:2]
            
            # ========== LOCAL DETECTION (EVERY FRAME - INSTANT!) ==========
            tracking_data = gaze_tracker.process_frame(frame)
            attention_result = attention_analyzer.analyze_frame(tracking_data)
            sound_details = sound_detector.get_detailed_status()
            
            # Verify face every 5 seconds
            identity_status, identity_color, match_conf = face_recognizer.get_status()
            if current_time - start_time > 5 and (current_time - start_time) % 5 < 0.1:
                face_recognizer.verify_face(frame)
                identity_status, identity_color, match_conf = face_recognizer.get_status()
            
            # ========== DISPLAY RESULTS (INSTANT FEEDBACK) ==========
            status = attention_result['status']
            candidate = attention_result['candidate_status']
            color = attention_result['color']
            details = attention_result['details']
            
            # Main status panel
            cv2.rectangle(frame, (0, 0), (w, 150), (0, 0, 0), -1)
            cv2.rectangle(frame, (0, 0), (w, 150), color, 3)
            
            cv2.putText(frame, f"Status: {status}", (20, 40),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
            cv2.putText(frame, candidate, (20, 75),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1)
            cv2.putText(frame, f"Identity: {identity_status}", (20, 110),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, identity_color, 2)
            
            # Tracking details (if available)
            if details:
                info_y = 160
                cv2.putText(frame, f"Head: Yaw={details['head_yaw']:.1f}° Pitch={details['head_pitch']:.1f}°", 
                           (20, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
                cv2.putText(frame, f"Gaze: X={details['gaze_dev']:.3f} Y={details['gaze_dev_y']:.3f}", 
                           (20, info_y+25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
                
                if details['yaw_compensating'] or details['pitch_compensating']:
                    cv2.putText(frame, "⚡ COMPENSATING", (20, info_y+50),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
            
            # Sound status
            sound_color = (0, 0, 255) if sound_details['is_detected'] else (100, 100, 100)
            sound_text = "🔊 SOUND" if sound_details['is_detected'] else "🔇 QUIET"
            cv2.putText(frame, sound_text, (w-200, 40),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, sound_color, 2)
            
            # Violation counter
            if status == "INATTENTIVE":
                violation_count += 1
            
            # Stats at bottom
            elapsed = current_time - start_time
            fps = frame_count / elapsed if elapsed > 0 else 0
            
            cv2.rectangle(frame, (0, h-70), (w, h), (0, 0, 0), -1)
            cv2.putText(frame, f"Time: {int(elapsed)}s | FPS: {fps:.1f}", 
                       (20, h-40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(frame, f"Violations: {violation_count} | Frames: {frame_count}", 
                       (20, h-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, 
                       (0, 0, 255) if violation_count > 0 else (0, 255, 0), 2)
            
            cv2.imshow("Real-time Proctoring", frame)
            
            # ========== ASYNC LOGGING TO API (doesn't slow down detection) ==========
            if session_id and current_time - last_log_time >= 1.0:
                try:
                    log_data = {
                        'timestamp': datetime.now().isoformat(),
                        'session_time': elapsed,
                        'attention_state': status,
                        'identity_status': identity_status,
                        'sound_detected': sound_details['is_detected'],
                        'sound_level': sound_details['rms_level'],
                        'head_yaw': details['head_yaw'] if details else 0.0,
                        'head_pitch': details['head_pitch'] if details else 0.0,
                        'gaze_x': tracking_data['gaze_x'] if tracking_data else 0.0,
                        'gaze_y': tracking_data['gaze_y'] if tracking_data else 0.0,
                        'ear': tracking_data['ear'] if tracking_data else 0.0,
                        'confidence': match_conf
                    }
                    log_queue.put_nowait(log_data)
                    last_log_time = current_time
                except queue.Full:
                    pass
            
            # Check for exit
            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC
                print("\n⚠ Monitoring stopped by user")
                break
            
    except KeyboardInterrupt:
        print("\n⚠ Interrupted by user")
    finally:
        # Cleanup
        print("\n" + "="*70)
        print("📊 FINAL STATISTICS")
        print("="*70)
        
        total_time = time.time() - start_time
        avg_fps = frame_count / total_time if total_time > 0 else 0
        violation_rate = (violation_count / frame_count * 100) if frame_count > 0 else 0
        
        print(f"Total time:           {total_time:.1f} seconds")
        print(f"Total frames:         {frame_count:,}")
        print(f"Average FPS:          {avg_fps:.1f}")
        print(f"Violations:           {violation_count:,} ({violation_rate:.1f}%)")
        print(f"Detection mode:       LOCAL (Real-time)")
        
        # Stop everything
        api_logger_running = False
        sound_detector.stop()
        gaze_tracker.close()
        cap.release()
        cv2.destroyAllWindows()
        
        # End API session
        if session_id:
            try:
                requests.delete(f"{API_URL}/api/sessions/{session_id}", timeout=2)
                print(f"✅ API session ended")
            except:
                pass
        
        print("="*70)
        print("✅ All systems stopped")
        print("="*70 + "\n")

if __name__ == "__main__":
    print("\n" + "="*70)
    print("REAL-TIME PROCTORING SYSTEM")
    print("="*70)
    print("\nThis version:")
    print("  ✅ Runs detection LOCALLY (every frame)")
    print("  ✅ Instant response (like your original main.py)")
    print("  ✅ Uses API only for session management")
    print("  ✅ Smooth 30 FPS camera")
    print("  ✅ Real-time violation detection")
    print("\nBest of both worlds!")
    print("="*70)
    
    main()