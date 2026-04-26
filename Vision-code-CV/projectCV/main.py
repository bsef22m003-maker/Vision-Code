"""
Main Entry Point for Exam Proctoring System
Complete modular implementation with ArcFace recognition
"""
import cv2
import time
from datetime import datetime

# Import all modules
from config import *
from sound_detector import SoundDetector
from face_recognizer import OpenCVFaceRecognizer as FaceRecognizer 
# for arcface we use from face_recognizer import ArcFaceRecognizer as FaceRecognizer 
from gaze_tracker import GazeTracker
from attention_analyzer import AttentionAnalyzer
from metrics import PerformanceMetrics
from data_logger import DataLogger
from ui_display import UIDisplay


def register_initial_face(cap, face_recognizer):
    """Register the user's face for identity verification"""
    print("\n" + "="*60)
    print("📸 FACE REGISTRATION (ArcFace Deep Learning)")  # Updated label
    print("="*60)
    print("Press SPACE to capture")
    print("="*60 + "\n")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            continue
        
        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]
        
        cv2.rectangle(frame, (10, 10), (w-10, h-10), (0, 255, 255), 3)
        cv2.putText(frame, "FACE REGISTRATION ", (w//2-200, 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
        cv2.putText(frame, "Center your face in the frame", (w//2-180, h-80), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(frame, "Press SPACE to register", (w//2-150, h-50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        cv2.imshow("Face Registration", frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == 32:  # Space
            success, message = face_recognizer.register_face(frame)
            print(f"{'✓' if success else '✗'} {message}")
            if success:
                cv2.destroyWindow("Face Registration")
                return True
            time.sleep(1)


def calibrate_attention(cap, gaze_tracker):
    """Calibrate baseline attention parameters"""
    print("\n" + "="*60)
    print("🎯 ATTENTION CALIBRATION")
    print("="*60)
    print(f"Look straight at the screen for {CALIBRATION_SECONDS} seconds...")
    print("="*60 + "\n")
    
    start = time.time()
    yaw_list, pitch_list, gx_list, gy_list = [], [], [], []
    
    while time.time() - start < CALIBRATION_SECONDS:
        ret, frame = cap.read()
        if not ret:
            continue
        
        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]
        
        tracking_data = gaze_tracker.process_frame(frame)
        
        if tracking_data is None:
            cv2.putText(frame, "No face - keep looking at camera", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.imshow("Calibrating...", frame)
            if cv2.waitKey(1) == 27:
                break
            continue
        
        yaw_list.append(tracking_data['yaw'] if tracking_data['yaw'] else 0.0)
        pitch_list.append(tracking_data['pitch'] if tracking_data['pitch'] else 0.0)
        gx_list.append(tracking_data['gaze_x'])
        gy_list.append(tracking_data['gaze_y'])

        elapsed = time.time() - start
        progress = int((elapsed / CALIBRATION_SECONDS) * 20)
        cv2.putText(frame, "Calibrating... keep eyes on the screen", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(frame, f"[{'=' * progress}{' ' * (20-progress)}] {elapsed:.1f}s", 
                   (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)
        
        try:
            cv2.circle(frame, tuple(tracking_data['left_iris'].astype(int)), 2, (0, 255, 255), -1)
            cv2.circle(frame, tuple(tracking_data['right_iris'].astype(int)), 2, (0, 255, 255), -1)
        except:
            pass
        
        cv2.imshow("Calibrating...", frame)
        if cv2.waitKey(1) == 27:
            break

    import numpy as np
    baseline_yaw = float(np.median(yaw_list)) if yaw_list else 0.0
    baseline_pitch = float(np.median(pitch_list)) if pitch_list else 0.0
    baseline_gaze_x = float(np.median(gx_list)) if gx_list else 0.5
    baseline_gaze_y = float(np.median(gy_list)) if gy_list else 0.5
    
    print(f"✓ Calibration complete:")
    print(f"  Yaw: {baseline_yaw:.2f}°, Pitch: {baseline_pitch:.2f}°")
    print(f"  Gaze: ({baseline_gaze_x:.3f}, {baseline_gaze_y:.3f})\n")
    
    cv2.destroyWindow("Calibrating...")
    return baseline_yaw, baseline_pitch, baseline_gaze_x, baseline_gaze_y


def main():
    """Main application loop"""
    # Initialize camera
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("❌ Camera not available")
    
    print("\n" + "="*60)
    print("🎓 EXAM PROCTORING SYSTEM (ArcFace Version)")  # Updated label
    print("="*60)
    print("Initializing all modules...")
    print("="*60 + "\n")
    
    # Initialize all modules
    sound_detector = SoundDetector()
    face_recognizer = FaceRecognizer() 
    gaze_tracker = GazeTracker()
    metrics = PerformanceMetrics()
    data_logger = DataLogger(CSV_FILENAME)
    ui_display = UIDisplay()
    
    # Start sound detection and calibrate
    sound_detector.start()
    time.sleep(1)
    sound_detector.auto_calibrate()
    
    # Register face
    register_initial_face(cap, face_recognizer)
    
    # Calibrate attention baseline
    baseline_yaw, baseline_pitch, baseline_gaze_x, baseline_gaze_y = calibrate_attention(cap, gaze_tracker)
    
    # Initialize attention analyzer with baseline
    attention_analyzer = AttentionAnalyzer(baseline_yaw, baseline_pitch, baseline_gaze_x, baseline_gaze_y)
    
    print("\n" + "="*60)
    print("🚀 SYSTEM ACTIVE - Monitoring started")
    print("="*60)
    print("Keyboard Commands:")
    print("  ESC → Exit")
    print("  'r' → Recalibrate attention")
    print("  'c' → Calibrate sound threshold")
    print("  's' → Show sound details")
    print("  '+' → Increase sound threshold")
    print("  '-' → Decrease sound threshold")
    print("="*60 + "\n")
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame = cv2.flip(frame, 1)
            
            # Get tracking data
            tracking_data = gaze_tracker.process_frame(frame)
            
            # Get sound status
            sound_details = sound_detector.get_detailed_status()
            metrics.update_sound(sound_details['is_detected'])
            
            # Verify identity periodically
            if hasattr(face_recognizer, 'reference_embedding'):
               check_attr = face_recognizer.reference_embedding
            elif hasattr(face_recognizer, 'reference_face'):
              check_attr = face_recognizer.reference_face
            else:
              check_attr = None
            if check_attr is not None:
                verify_result, verify_msg = face_recognizer.verify_face(frame)
                if verify_result is not None:
                    metrics.update_identity(verify_result)
                    print(f"Identity check: {verify_msg}")
            
            identity_status, identity_color, match_conf = face_recognizer.get_status()
            
            # Analyze attention
            attention_result = attention_analyzer.analyze_frame(tracking_data)
            
            # Update metrics
            predicted_attentive = attention_result['status'] == "ATTENTIVE"
            actual_attentive = attention_result['candidate_status'].startswith("ATTENTIVE")
            metrics.update(predicted_attentive, actual_attentive)
            
            # Print periodic metrics
            metrics.print_metrics()
            
            # Log data
            session_time = ui_display.get_session_time()
            data_logger.log_data({
                'timestamp': datetime.now().isoformat(),
                'session_time': round(session_time, 2),
                'attention_state': attention_result['status'],
                'identity_status': identity_status,
                'sound_detected': sound_details['is_detected'],
                'sound_level': round(sound_details['rms_level'], 4),
                'head_yaw': round(attention_result['details']['head_yaw'], 2) if attention_result['details'] else 0.0,
                'head_pitch': round(attention_result['details']['head_pitch'], 2) if attention_result['details'] else 0.0,
                'gaze_x': round(tracking_data['gaze_x'], 3) if tracking_data else 0.0,
                'gaze_y': round(tracking_data['gaze_y'], 3) if tracking_data else 0.0,
                'ear': round(tracking_data['ear'], 3) if tracking_data else 0.0,
                'confidence': round(match_conf, 2)
            })
            
            # Render UI
            frame = ui_display.render_frame(
                frame, attention_result, tracking_data,
                identity_status, identity_color, sound_details,
                face_recognizer.reference_image
            )
            
            cv2.imshow("Exam Proctoring System", frame)
            
            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC
                break
            elif key == ord('r'):
                print("\n🔄 Recalibrating attention...")
                baseline_yaw, baseline_pitch, baseline_gaze_x, baseline_gaze_y = calibrate_attention(cap, gaze_tracker)
                attention_analyzer.update_baseline(baseline_yaw, baseline_pitch, baseline_gaze_x, baseline_gaze_y)
            elif key == ord('c'):
                print("\n🔊 Calibrating sound...")
                sound_detector.auto_calibrate(duration=3.0)
            elif key == ord('s'):
                details = sound_detector.get_detailed_status()
                print(f"\n📊 Sound Status:")
                print(f"   Detected: {details['is_detected']}")
                print(f"   RMS Level: {details['rms_level']:.4f}")
                print(f"   Peak Level: {details['peak_level']:.4f}")
                print(f"   Threshold: {details['threshold']:.4f}")
                print(f"   Sensitivity: {details['sensitivity_percent']:.1f}%\n")
            elif key == ord('+') or key == ord('='):
                sound_detector.adjust_threshold(0.005)
            elif key == ord('-') or key == ord('_'):
                sound_detector.adjust_threshold(-0.005)

    except KeyboardInterrupt:
        print("\n⚠️ Interrupted by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("\n" + "="*60)
        print("🛑 SHUTTING DOWN")
        print("="*60)
        
        # Cleanup
        sound_detector.stop()
        data_logger.finalize()
        gaze_tracker.close()
        cap.release()
        cv2.destroyAllWindows()
        
        # Print final report
        print("\n" + metrics.get_summary_report())
        
        session_time = ui_display.get_session_time()
        frame_count = ui_display.get_frame_count()
        
        print(f"✓ CSV saved: {CSV_FILENAME}")
        print("✓ All systems stopped")
        print("="*60)
        print("\n📊 Session Summary:")
        print(f"   Duration: {int(session_time//60)}m {int(session_time%60)}s")
        print(f"   Total Frames: {frame_count}")
        print(f"   Data Points Logged: {frame_count}")
        print("\nThank you for using the Exam Proctoring System!")
        print("="*60 + "\n")


if __name__ == "__main__":
    main()