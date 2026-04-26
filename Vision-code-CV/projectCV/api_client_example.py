"""
Example Python client for the Exam Proctoring API
Demonstrates how to use the API endpoints
"""
import requests
import cv2
import base64
import time
import json

# API Configuration
API_BASE_URL = "http://localhost:8000"

class ProctoringClient:
    def __init__(self, base_url=API_BASE_URL):
        self.base_url = base_url
        self.session_id = None
        
    def check_health(self):
        """Check if API is online"""
        response = requests.get(f"{self.base_url}/")
        return response.json()
    
    def create_session(self, user_id, exam_id, use_arcface=False):
        """Create a new proctoring session"""
        data = {
            "user_id": user_id,
            "exam_id": exam_id,
            "use_arcface": use_arcface,
            "sound_threshold": 0.01
        }
        response = requests.post(f"{self.base_url}/api/sessions/create", json=data)
        result = response.json()
        self.session_id = result['session_id']
        print(f"✓ Session created: {self.session_id}")
        return result
    
    def register_face_from_file(self, image_path):
        """Register face from image file"""
        if not self.session_id:
            raise Exception("No active session. Create session first.")
        
        with open(image_path, 'rb') as f:
            files = {'file': ('image.jpg', f, 'image/jpeg')}
            response = requests.post(
                f"{self.base_url}/api/sessions/{self.session_id}/register-face",
                files=files
            )
        result = response.json()
        print(f"{'✓' if result['success'] else '✗'} Face registration: {result['message']}")
        return result
    
    def register_face_from_camera(self):
        """Register face by capturing from camera"""
        if not self.session_id:
            raise Exception("No active session. Create session first.")
        
        cap = cv2.VideoCapture(0)
        print("Press SPACE to capture face for registration...")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                continue
            
            frame = cv2.flip(frame, 1)
            cv2.putText(frame, "Press SPACE to register face", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.imshow("Register Face", frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == 32:  # Space
                # Encode frame to base64
                _, buffer = cv2.imencode('.jpg', frame)
                img_base64 = base64.b64encode(buffer).decode('utf-8')
                
                # Send to API
                response = requests.post(
                    f"{self.base_url}/api/sessions/{self.session_id}/register-face-base64",
                    json={"image": f"data:image/jpeg;base64,{img_base64}"}
                )
                result = response.json()
                print(f"\n{'✓' if result['success'] else '✗'} {result['message']}")
                
                if result['success']:
                    cap.release()
                    cv2.destroyAllWindows()
                    return result
                
            elif key == 27:  # ESC
                break
        
        cap.release()
        cv2.destroyAllWindows()
        return None
    
    def calibrate_from_camera(self, duration=3.0):
        """Calibrate attention baseline from camera"""
        if not self.session_id:
            raise Exception("No active session. Create session first.")
        
        cap = cv2.VideoCapture(0)
        
        # Import gaze tracker locally to get baseline
        from gaze_tracker import GazeTracker
        gaze_tracker = GazeTracker()
        
        print(f"Calibrating... Look at the screen for {duration} seconds")
        
        yaw_list, pitch_list, gx_list, gy_list = [], [], [], []
        start_time = time.time()
        
        while time.time() - start_time < duration:
            ret, frame = cap.read()
            if not ret:
                continue
            
            frame = cv2.flip(frame, 1)
            tracking_data = gaze_tracker.process_frame(frame)
            
            if tracking_data:
                yaw_list.append(tracking_data['yaw'] or 0.0)
                pitch_list.append(tracking_data['pitch'] or 0.0)
                gx_list.append(tracking_data['gaze_x'])
                gy_list.append(tracking_data['gaze_y'])
            
            elapsed = time.time() - start_time
            cv2.putText(frame, f"Calibrating... {elapsed:.1f}s", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.imshow("Calibration", frame)
            cv2.waitKey(1)
        
        cap.release()
        cv2.destroyAllWindows()
        gaze_tracker.close()
        
        # Calculate baseline
        import numpy as np
        baseline_yaw = float(np.median(yaw_list)) if yaw_list else 0.0
        baseline_pitch = float(np.median(pitch_list)) if pitch_list else 0.0
        baseline_gaze_x = float(np.median(gx_list)) if gx_list else 0.5
        baseline_gaze_y = float(np.median(gy_list)) if gy_list else 0.5
        
        # Send calibration to API
        calibration_data = {
            "baseline_yaw": baseline_yaw,
            "baseline_pitch": baseline_pitch,
            "baseline_gaze_x": baseline_gaze_x,
            "baseline_gaze_y": baseline_gaze_y
        }
        
        response = requests.post(
            f"{self.base_url}/api/sessions/{self.session_id}/calibrate",
            json=calibration_data
        )
        result = response.json()
        print(f"✓ Calibration complete: Yaw={baseline_yaw:.2f}°, Pitch={baseline_pitch:.2f}°")
        return result
    
    def analyze_frame_from_camera(self):
        """Capture and analyze frame from camera"""
        if not self.session_id:
            raise Exception("No active session. Create session first.")
        
        cap = cv2.VideoCapture(0)
        ret, frame = cap.read()
        cap.release()
        
        if not ret:
            return None
        
        frame = cv2.flip(frame, 1)
        
        # Encode to base64
        _, buffer = cv2.imencode('.jpg', frame)
        img_base64 = base64.b64encode(buffer).decode('utf-8')
        
        # Send to API
        response = requests.post(
            f"{self.base_url}/api/sessions/{self.session_id}/analyze-frame-base64",
            json={"image": f"data:image/jpeg;base64,{img_base64}"}
        )
        
        return response.json()
    
    def start_live_monitoring(self, duration_seconds=60):
        """Start live monitoring from camera"""
        if not self.session_id:
            raise Exception("No active session. Create session first.")
        
        cap = cv2.VideoCapture(0)
        start_time = time.time()
        frame_count = 0
        
        print(f"\n{'='*60}")
        print(f"🎥 LIVE MONITORING STARTED (Duration: {duration_seconds}s)")
        print(f"{'='*60}\n")
        
        try:
            while time.time() - start_time < duration_seconds:
                ret, frame = cap.read()
                if not ret:
                    continue
                
                frame = cv2.flip(frame, 1)
                frame_count += 1
                
                # Analyze every frame (you may want to throttle this)
                if frame_count % 5 == 0:  # Analyze every 5th frame
                    _, buffer = cv2.imencode('.jpg', frame)
                    img_base64 = base64.b64encode(buffer).decode('utf-8')
                    
                    try:
                        response = requests.post(
                            f"{self.base_url}/api/sessions/{self.session_id}/analyze-frame-base64",
                            json={"image": f"data:image/jpeg;base64,{img_base64}"},
                            timeout=2
                        )
                        result = response.json()
                        
                        # Display results on frame
                        status = result['attention_status']
                        color = tuple(result['attention_color'])
                        
                        cv2.putText(frame, f"Attention: {status}", (10, 30),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                        cv2.putText(frame, f"Identity: {result['identity_status']}", (10, 60),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)
                        
                        # Display warnings
                        if result['warnings']:
                            y_pos = 90
                            for warning in result['warnings']:
                                cv2.putText(frame, f"⚠ {warning}", (10, y_pos),
                                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                                y_pos += 25
                        
                        # Print to console
                        if result['warnings']:
                            print(f"[{frame_count:04d}] ⚠ VIOLATIONS: {', '.join(result['warnings'])}")
                        
                    except requests.exceptions.Timeout:
                        print(f"[{frame_count:04d}] ⚠ Request timeout")
                    except Exception as e:
                        print(f"[{frame_count:04d}] ✗ Error: {e}")
                
                # Display frame
                elapsed = time.time() - start_time
                cv2.putText(frame, f"Time: {int(elapsed)}s / {duration_seconds}s", 
                           (10, frame.shape[0] - 20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)
                cv2.imshow("Live Monitoring", frame)
                
                if cv2.waitKey(1) & 0xFF == 27:  # ESC to exit
                    break
            
        finally:
            cap.release()
            cv2.destroyAllWindows()
            
            print(f"\n{'='*60}")
            print(f"📊 Monitoring Complete - Frames Processed: {frame_count}")
            print(f"{'='*60}\n")
    
    def get_metrics(self):
        """Get session metrics"""
        if not self.session_id:
            raise Exception("No active session")
        
        response = requests.get(f"{self.base_url}/api/sessions/{self.session_id}/metrics")
        return response.json()
    
    def get_status(self):
        """Get session status"""
        if not self.session_id:
            raise Exception("No active session")
        
        response = requests.get(f"{self.base_url}/api/sessions/{self.session_id}/status")
        return response.json()
    
    def end_session(self):
        """End the session"""
        if not self.session_id:
            raise Exception("No active session")
        
        response = requests.delete(f"{self.base_url}/api/sessions/{self.session_id}")
        result = response.json()
        print(f"✓ Session ended: {self.session_id}")
        self.session_id = None
        return result


# ==================== Example Usage ====================

def main():
    """Complete workflow example"""
    
    # Initialize client
    client = ProctoringClient()
    
    # 1. Check API health
    print("Checking API health...")
    health = client.check_health()
    print(f"✓ API Status: {health['status']}")
    print()
    
    # 2. Create session
    session = client.create_session(
        user_id="student123",
        exam_id="final_exam_2024",
        use_arcface=False  # Set to True if you have InsightFace installed
    )
    print()
    
    # 3. Register face
    print("Registering face...")
    client.register_face_from_camera()
    print()
    
    # 4. Calibrate
    print("Calibrating attention baseline...")
    client.calibrate_from_camera(duration=3.0)
    print()
    
    # 5. Start live monitoring
    client.start_live_monitoring(duration_seconds=30)
    
    # 6. Get final metrics
    print("Fetching session metrics...")
    metrics = client.get_metrics()
    print(f"\n📊 Session Metrics:")
    print(f"   Total Frames: {metrics['total_frames']}")
    print(f"   Attention Rate: {metrics['attention_rate']*100:.1f}%")
    print(f"   Accuracy: {metrics['accuracy']*100:.1f}%")
    print(f"   Sound Detections: {metrics['sound_detections']}")
    print()
    
    # 7. End session
    client.end_session()


if __name__ == "__main__":
    main()