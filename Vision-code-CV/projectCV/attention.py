"""
Complete Attention Tracker with OpenCV Face Recognition
✓ OpenCV-based face detection (no face_recognition library needed)
✓ Red color for sound detection
✓ Real-time performance metrics every 30 seconds
✓ Comprehensive end-of-session report
✓ Full confusion matrix tracking
"""

import cv2
import mediapipe as mp
import numpy as np
import time
import sounddevice as sd
from collections import deque
import threading
import csv
from datetime import datetime
import os

# ----------------- Config -----------------
CALIBRATION_SECONDS = 2.0
FACE_VERIFY_INTERVAL = 5.0

# Sound Detection Config
SOUND_THRESHOLD = 0.01
SOUND_SAMPLE_RATE = 16000
SOUND_BLOCK_SIZE = 1024
SOUND_SMOOTHING = 20

# Model points for solvePnP
MODEL_POINTS = np.array([
    (0.0, 0.0, 0.0),
    (0.0, -63.6, -12.5),
    (-43.3, 32.7, -26.0),
    (43.3, 32.7, -26.0),
    (-28.9, 28.9, -24.1),
    (28.9, 28.9, -24.1)
], dtype=np.float64)

POSE_IDX = [1, 152, 33, 263, 61, 291]
LEFT_IRIS_IDX = [468, 469, 470, 471]
RIGHT_IRIS_IDX = [473, 474, 475, 476]
LEFT_EYE_EAR_IDX = [33, 160, 158, 133, 153, 144]
RIGHT_EYE_EAR_IDX = [362, 385, 387, 263, 373, 380]
LEFT_EYE_CORNERS = (33, 133)
RIGHT_EYE_CORNERS = (362, 263)
LEFT_EYE_TOP_BOTTOM = (159, 145)
RIGHT_EYE_TOP_BOTTOM = (386, 374)

# Thresholds
HEAD_YAW_TOL_DEG = 25.0
HEAD_PITCH_TOL_DEG = 22.0
GAZE_THRESH = 0.07
GAZE_RELAXED = 0.12
COMPENSATION_MIN = 0.003
COMPENSATION_PITCH_MIN = 0.015
EAR_THRESH = 0.20
EYE_CLOSED_CONSEC = 3
SMOOTH_WINDOW = 7
HOLD_FRAMES = 6
MAX_GAZE_DEVIATION = 0.20

# CSV Logging
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_FILENAME = os.path.join(SCRIPT_DIR, f"attention_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
CSV_WRITE_INTERVAL = 1.0

# Performance Metrics
METRICS_UPDATE_INTERVAL = 30.0

# ----------------- Performance Metrics Class -----------------
class PerformanceMetrics:
    def __init__(self):
        # Confusion matrix
        self.true_positive = 0
        self.true_negative = 0
        self.false_positive = 0
        self.false_negative = 0
        
        # Tracking
        self.total_frames = 0
        self.attentive_frames = 0
        self.inattentive_frames = 0
        self.sound_detected_count = 0
        self.face_verified_count = 0
        self.face_failed_count = 0
        
        self.last_metrics_print = time.time()
        
    def update(self, predicted_attentive, actual_attentive):
        """Update confusion matrix"""
        self.total_frames += 1
        
        if predicted_attentive:
            self.attentive_frames += 1
            if actual_attentive:
                self.true_positive += 1
            else:
                self.false_positive += 1
        else:
            self.inattentive_frames += 1
            if actual_attentive:
                self.false_negative += 1
            else:
                self.true_negative += 1
    
    def update_sound(self, detected):
        if detected:
            self.sound_detected_count += 1
    
    def update_identity(self, verified):
        if verified:
            self.face_verified_count += 1
        else:
            self.face_failed_count += 1
    
    def calculate_metrics(self):
        """Calculate precision, recall, F1-score, accuracy"""
        precision = self.true_positive / (self.true_positive + self.false_positive) \
                   if (self.true_positive + self.false_positive) > 0 else 0.0
        
        recall = self.true_positive / (self.true_positive + self.false_negative) \
                if (self.true_positive + self.false_negative) > 0 else 0.0
        
        f1_score = 2 * (precision * recall) / (precision + recall) \
                  if (precision + recall) > 0 else 0.0
        
        accuracy = (self.true_positive + self.true_negative) / self.total_frames \
                  if self.total_frames > 0 else 0.0
        
        return {
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score,
            'accuracy': accuracy,
            'true_positive': self.true_positive,
            'true_negative': self.true_negative,
            'false_positive': self.false_positive,
            'false_negative': self.false_negative
        }
    
    def print_metrics(self, force=False):
        """Print metrics every interval"""
        current_time = time.time()
        
        if force or (current_time - self.last_metrics_print >= METRICS_UPDATE_INTERVAL):
            metrics = self.calculate_metrics()
            
            print("\n" + "="*70)
            print("📊 PERFORMANCE METRICS")
            print("="*70)
            print(f"Total Frames Processed:     {self.total_frames:,}")
            print(f"Attentive Frames:           {self.attentive_frames:,} ({self.attentive_frames/self.total_frames*100:.1f}%)")
            print(f"Inattentive Frames:         {self.inattentive_frames:,} ({self.inattentive_frames/self.total_frames*100:.1f}%)")
            print("-"*70)
            print("Confusion Matrix:")
            print(f"  True Positives (TP):      {metrics['true_positive']:,}")
            print(f"  True Negatives (TN):      {metrics['true_negative']:,}")
            print(f"  False Positives (FP):     {metrics['false_positive']:,}")
            print(f"  False Negatives (FN):     {metrics['false_negative']:,}")
            print("-"*70)
            print(f"✓ Precision:                {metrics['precision']:.4f} ({metrics['precision']*100:.2f}%)")
            print(f"✓ Recall:                   {metrics['recall']:.4f} ({metrics['recall']*100:.2f}%)")
            print(f"✓ F1-Score:                 {metrics['f1_score']:.4f} ({metrics['f1_score']*100:.2f}%)")
            print(f"✓ Accuracy:                 {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
            print("-"*70)
            print(f"Sound Detections:           {self.sound_detected_count:,}")
            print(f"Face Verifications (Pass):  {self.face_verified_count:,}")
            print(f"Face Verifications (Fail):  {self.face_failed_count:,}")
            print("="*70 + "\n")
            
            self.last_metrics_print = current_time
            return metrics
        
        return None
    
    def get_summary_report(self):
        """Generate comprehensive end-of-session report"""
        metrics = self.calculate_metrics()
        
        report = f"""
╔══════════════════════════════════════════════════════════════════════╗
║                    FINAL SESSION PERFORMANCE REPORT                  ║
╚══════════════════════════════════════════════════════════════════════╝

📊 OVERALL STATISTICS
{'─'*70}
Total Frames Analyzed:          {self.total_frames:,}
Session Attention Rate:         {self.attentive_frames/self.total_frames*100:.2f}%
Session Inattention Rate:       {self.inattentive_frames/self.total_frames*100:.2f}%

🎯 CLASSIFICATION METRICS
{'─'*70}
Precision:                      {metrics['precision']:.4f} ({metrics['precision']*100:.2f}%)
  → Of predicted attentive, {metrics['precision']*100:.1f}% were actually attentive

Recall (Sensitivity):           {metrics['recall']:.4f} ({metrics['recall']*100:.2f}%)
  → Of actual attentive states, {metrics['recall']*100:.1f}% were detected

F1-Score:                       {metrics['f1_score']:.4f} ({metrics['f1_score']*100:.2f}%)
  → Harmonic mean of precision and recall

Accuracy:                       {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)
  → Overall correctness of predictions

📈 CONFUSION MATRIX
{'─'*70}
                    Predicted Attentive  |  Predicted Inattentive
Actual Attentive       {metrics['true_positive']:6,}        |       {metrics['false_negative']:6,}
Actual Inattentive     {metrics['false_positive']:6,}        |       {metrics['true_negative']:6,}

🔊 BEHAVIORAL EVENTS
{'─'*70}
Sound Detections:               {self.sound_detected_count:,} times
Face Verifications (Pass):      {self.face_verified_count:,} times
Face Verifications (Fail):      {self.face_failed_count:,} times

💡 PERFORMANCE INTERPRETATION
{'─'*70}
"""
        if metrics['accuracy'] >= 0.9:
            report += "Excellent: System performed with high accuracy (≥90%)\n"
        elif metrics['accuracy'] >= 0.8:
            report += "Good: System performed well (80-90%)\n"
        elif metrics['accuracy'] >= 0.7:
            report += "Fair: System performed adequately (70-80%)\n"
        else:
            report += "Poor: System accuracy below 70% - consider recalibration\n"
        
        if metrics['f1_score'] >= 0.85:
            report += "Balanced: Good balance between precision and recall\n"
        else:
            report += "Unbalanced: Consider adjusting detection thresholds\n"
        
        report += "╚" + "═"*70 + "╝\n"
        
        return report

# ----------------- CSV Logger -----------------
class DataLogger:
    def __init__(self, filename):
        self.filename = filename
        self.data_buffer = []
        self.last_write_time = time.time()
        
        with open(self.filename, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'Timestamp', 'Session_Time_Seconds', 'Attention_State',
                'Identity_Status', 'Sound_Detected', 'Sound_Level',
                'Head_Yaw', 'Head_Pitch', 'Gaze_X', 'Gaze_Y',
                'Eye_Aspect_Ratio', 'Confidence'
            ])
        print(f"✓ CSV logging initialized: {self.filename}")
    
    def log_data(self, data_dict):
        self.data_buffer.append(data_dict)
        if time.time() - self.last_write_time >= CSV_WRITE_INTERVAL:
            self.write_buffer()
            self.last_write_time = time.time()
    
    def write_buffer(self):
        if not self.data_buffer:
            return
        try:
            with open(self.filename, 'a', newline='') as f:
                writer = csv.writer(f)
                for data in self.data_buffer:
                    writer.writerow([
                        data.get('timestamp', ''), data.get('session_time', 0),
                        data.get('attention_state', 'UNKNOWN'), data.get('identity_status', 'UNKNOWN'),
                        data.get('sound_detected', False), data.get('sound_level', 0.0),
                        data.get('head_yaw', 0.0), data.get('head_pitch', 0.0),
                        data.get('gaze_x', 0.0), data.get('gaze_y', 0.0),
                        data.get('ear', 0.0), data.get('confidence', 0.0)
                    ])
            self.data_buffer.clear()
        except Exception as e:
            print(f"Error writing CSV: {e}")
    
    def finalize(self):
        self.write_buffer()
        print(f"✓ CSV completed: {self.filename}")

# ----------------- OpenCV Face Recognition -----------------
class OpenCVFaceRecognizer:
    def __init__(self):
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        self.face_cascade = cv2.CascadeClassifier(cascade_path)
        
        self.reference_face = None
        self.reference_histogram = None
        self.reference_image = None  # Store BGR image for display
        self.last_check_time = 0
        self.identity_status = "NOT REGISTERED"
        self.identity_color = (128, 128, 128)
        self.match_confidence = 0.0
        
        print("✓ OpenCV face detector initialized")
        
    def register_face(self, frame):
        """Register reference face using OpenCV"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.3, 5, minSize=(100, 100))
        
        if len(faces) == 0:
            return False, "No face detected"
        
        if len(faces) > 1:
            return False, "Multiple faces detected - ensure only one person"
        
        x, y, w, h = faces[0]
        face_roi = gray[y:y+h, x:x+w]
        face_roi_resized = cv2.resize(face_roi, (200, 200))
        
        self.reference_face = face_roi_resized
        self.reference_histogram = cv2.calcHist([face_roi_resized], [0], None, [256], [0, 256])
        cv2.normalize(self.reference_histogram, self.reference_histogram)
        
        # Store colored version for thumbnail display
        self.reference_image = frame.copy()
        
        self.identity_status = "VERIFIED"
        self.identity_color = (0, 255, 0)
        
        return True, "Face registered successfully"
    
    def verify_face(self, frame):
        """Verify face using histogram comparison"""
        if self.reference_face is None:
            return False, "No reference face"
        
        current_time = time.time()
        if current_time - self.last_check_time < FACE_VERIFY_INTERVAL:
            return None, "Waiting"
        
        self.last_check_time = current_time
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.3, 5, minSize=(100, 100))
        
        if len(faces) == 0:
            self.identity_status = "NO FACE"
            self.identity_color = (0, 0, 255)
            return False, "No face detected"
        
        if len(faces) > 1:
            self.identity_status = "MULTIPLE FACES"
            self.identity_color = (0, 0, 255)
            return False, "Multiple faces"
        
        x, y, w, h = faces[0]
        face_roi = gray[y:y+h, x:x+w]
        face_roi_resized = cv2.resize(face_roi, (200, 200))
        
        current_hist = cv2.calcHist([face_roi_resized], [0], None, [256], [0, 256])
        cv2.normalize(current_hist, current_hist)
        
        similarity = cv2.compareHist(self.reference_histogram, current_hist, cv2.HISTCMP_CORREL)
        self.match_confidence = float(similarity)
        
        if similarity > 0.7:
            self.identity_status = "VERIFIED"
            self.identity_color = (0, 255, 0)
            return True, f"Match ({similarity:.2f})"
        else:
            self.identity_status = "DIFFERENT PERSON"
            self.identity_color = (0, 0, 255)
            return False, f"No match ({similarity:.2f})"
    
    def get_status(self):
        return self.identity_status, self.identity_color, self.match_confidence

# ----------------- Sound Detector -----------------
class SoundDetector:
    def __init__(self, threshold=SOUND_THRESHOLD, sample_rate=SOUND_SAMPLE_RATE):
        self.threshold = threshold
        self.sample_rate = sample_rate
        self.sound_levels = deque(maxlen=SOUND_SMOOTHING)
        self.is_sound_detected = False
        self.current_level = 0.0
        self.running = False
        self.stream = None
        self.lock = threading.Lock()
        self.peak_level = 0.0
        self.peak_history = deque(maxlen=10)
        
    def audio_callback(self, indata, frames, time_info, status):
        if status and 'overflow' not in str(status).lower():
            print(f"Sound status: {status}")
        
        try:
            rms = np.sqrt(np.mean(indata**2))
            peak = np.max(np.abs(indata))
            
            with self.lock:
                self.sound_levels.append(float(rms))
                self.peak_history.append(float(peak))
                self.current_level = float(np.mean(self.sound_levels))
                self.peak_level = float(np.max(self.peak_history))
                avg_exceeded = self.current_level > self.threshold
                peak_exceeded = self.peak_level > (self.threshold * 2.5)
                self.is_sound_detected = avg_exceeded or peak_exceeded
        except Exception as e:
            print(f"Audio error: {e}")
    
    def start(self):
        self.running = True
        try:
            device_info = sd.query_devices(kind='input')
            print(f"✓ Audio device: {device_info['name']}")
            
            self.stream = sd.InputStream(
                callback=self.audio_callback,
                channels=1,
                samplerate=self.sample_rate,
                blocksize=SOUND_BLOCK_SIZE,
                dtype='float32',
                latency='low'
            )
            self.stream.start()
            print(f"✓ Sound detection started (threshold: {self.threshold:.4f})")
        except Exception as e:
            print(f"✗ Sound error: {e}")
            try:
                self.stream = sd.InputStream(
                    callback=self.audio_callback,
                    channels=1,
                    samplerate=8000,
                    blocksize=512,
                    dtype='float32'
                )
                self.stream.start()
                self.sample_rate = 8000
                print("✓ Sound detection started (fallback)")
            except:
                self.running = False
    
    def stop(self):
        if self.stream:
            self.stream.stop()
            self.stream.close()
    
    def get_detailed_status(self):
        with self.lock:
            sensitivity = (self.current_level / self.threshold * 100) if self.threshold > 0 else 0
            return {
                'is_detected': self.is_sound_detected,
                'rms_level': self.current_level,
                'peak_level': self.peak_level,
                'threshold': self.threshold,
                'sensitivity_percent': sensitivity
            }
    
    def adjust_threshold(self, delta):
        with self.lock:
            old = self.threshold
            self.threshold = max(0.001, min(0.5, self.threshold + delta))
            print(f"Threshold: {old:.4f} → {self.threshold:.4f}")
    
    def auto_calibrate(self, duration=3.0):
        print(f"\n🔊 Auto-calibrating sound...")
        print(f"   Stay SILENT for {duration} seconds...")
        
        calibration_samples = []
        start_time = time.time()
        
        while time.time() - start_time < duration:
            with self.lock:
                if len(self.sound_levels) > 0:
                    calibration_samples.append(self.current_level)
            time.sleep(0.1)
            elapsed = time.time() - start_time
            progress = int((elapsed / duration) * 20)
            print(f"\r   [{'=' * progress}{' ' * (20-progress)}] {elapsed:.1f}s", end='')
        
        print()
        
        if calibration_samples:
            ambient_noise = np.mean(calibration_samples)
            noise_std = np.std(calibration_samples)
            suggested_threshold = ambient_noise + (3 * noise_std)
            suggested_threshold = max(0.005, min(0.1, suggested_threshold))
            
            with self.lock:
                self.threshold = suggested_threshold
            
            print(f"   ✓ Calibration complete! Threshold: {self.threshold:.4f}\n")

# ----------------- MediaPipe Setup -----------------
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

def solve_head_pose(landmarks, img_w, img_h):
    image_points = np.array([
        (landmarks[POSE_IDX[i]].x * img_w, landmarks[POSE_IDX[i]].y * img_h)
        for i in range(6)
    ], dtype=np.float64)

    focal_length = img_w
    cam_matrix = np.array([[focal_length, 0, img_w/2],
                           [0, focal_length, img_h/2],
                           [0, 0, 1]], dtype=np.float64)
    dist_coeffs = np.zeros((4,1))

    success, rot_vec, _ = cv2.solvePnP(MODEL_POINTS, image_points, cam_matrix, 
                                       dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)
    if not success:
        return None, None
    
    rmat, _ = cv2.Rodrigues(rot_vec)
    sy = np.sqrt(rmat[0,0]**2 + rmat[1,0]**2)
    
    if sy < 1e-6:
        x = np.arctan2(-rmat[1,2], rmat[1,1])
        y = np.arctan2(-rmat[2,0], sy)
    else:
        x = np.arctan2(rmat[2,1], rmat[2,2])
        y = np.arctan2(-rmat[2,0], sy)
    
    return np.degrees(y), np.degrees(x)

def iris_centers(landmarks, w, h):
    try:
        left_pts = np.array([(landmarks[i].x * w, landmarks[i].y * h) for i in LEFT_IRIS_IDX])
        right_pts = np.array([(landmarks[i].x * w, landmarks[i].y * h) for i in RIGHT_IRIS_IDX])
        return np.mean(left_pts, axis=0), np.mean(right_pts, axis=0)
    except:
        return None, None

def eye_aspect_ratio(landmarks, eye_idx, w, h):
    pts = [(int(landmarks[i].x * w), int(landmarks[i].y * h)) for i in eye_idx]
    p = np.array(pts, dtype=np.float32)
    A = np.linalg.norm(p[1] - p[5])
    B = np.linalg.norm(p[2] - p[4])
    C = np.linalg.norm(p[0] - p[3])
    return float((A + B) / (2.0 * C)) if C > 1e-6 else 0.0

# ----------------- Registration -----------------
def register_initial_face(cap, face_recognizer):
    print("\n" + "="*60)
    print("📸 FACE REGISTRATION (OpenCV Method)")
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
        cv2.putText(frame, "FACE REGISTRATION", (w//2-150, 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
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

def calibrate(cap):
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
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = face_mesh.process(rgb)
        
        if not res.multi_face_landmarks:
            cv2.putText(frame, "No face - keep looking at camera", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
            cv2.imshow("Calibrating...", frame)
            if cv2.waitKey(1) == 27:
                break
            continue
        
        lm = res.multi_face_landmarks[0].landmark
        yaw, pitch = solve_head_pose(lm, w, h)
        left_iris, right_iris = iris_centers(lm, w, h)
        
        if left_iris is None:
            continue
        
        left_eye_left = np.array([lm[LEFT_EYE_CORNERS[0]].x * w, lm[LEFT_EYE_CORNERS[0]].y * h])
        left_eye_right = np.array([lm[LEFT_EYE_CORNERS[1]].x * w, lm[LEFT_EYE_CORNERS[1]].y * h])
        right_eye_left = np.array([lm[RIGHT_EYE_CORNERS[0]].x * w, lm[RIGHT_EYE_CORNERS[0]].y * h])
        right_eye_right = np.array([lm[RIGHT_EYE_CORNERS[1]].x * w, lm[RIGHT_EYE_CORNERS[1]].y * h])
        
        left_top = np.array([lm[LEFT_EYE_TOP_BOTTOM[0]].x * w, lm[LEFT_EYE_TOP_BOTTOM[0]].y * h])
        left_bottom = np.array([lm[LEFT_EYE_TOP_BOTTOM[1]].x * w, lm[LEFT_EYE_TOP_BOTTOM[1]].y * h])
        right_top = np.array([lm[RIGHT_EYE_TOP_BOTTOM[0]].x * w, lm[RIGHT_EYE_TOP_BOTTOM[0]].y * h])
        right_bottom = np.array([lm[RIGHT_EYE_TOP_BOTTOM[1]].x * w, lm[RIGHT_EYE_TOP_BOTTOM[1]].y * h])
        
        left_gx = (left_iris[0] - left_eye_left[0]) / (left_eye_right[0] - left_eye_left[0] + 1e-6)
        right_gx = (right_iris[0] - right_eye_left[0]) / (right_eye_right[0] - right_eye_left[0] + 1e-6)
        left_gy = (left_iris[1] - left_top[1]) / (left_bottom[1] - left_top[1] + 1e-6)
        right_gy = (right_iris[1] - right_top[1]) / (right_bottom[1] - right_top[1] + 1e-6)
        
        avg_gx = (left_gx + right_gx) / 2.0
        avg_gy = (left_gy + right_gy) / 2.0
        
        yaw_list.append(yaw if yaw else 0.0)
        pitch_list.append(pitch if pitch else 0.0)
        gx_list.append(avg_gx)
        gy_list.append(avg_gy)

        elapsed = time.time() - start
        progress = int((elapsed / CALIBRATION_SECONDS) * 20)
        cv2.putText(frame, "Calibrating... keep eyes on the screen", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)
        cv2.putText(frame, f"[{'=' * progress}{' ' * (20-progress)}] {elapsed:.1f}s", 
                   (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 1)
        
        try:
            cv2.circle(frame, tuple(left_iris.astype(int)), 2, (0,255,255), -1)
            cv2.circle(frame, tuple(right_iris.astype(int)), 2, (0,255,255), -1)
        except:
            pass
        
        cv2.imshow("Calibrating...", frame)
        if cv2.waitKey(1) == 27:
            break

    baseline_yaw = float(np.median(yaw_list)) if yaw_list else 0.0
    baseline_pitch = float(np.median(pitch_list)) if pitch_list else 0.0
    baseline_gaze_x = float(np.median(gx_list)) if gx_list else 0.5
    baseline_gaze_y = float(np.median(gy_list)) if gy_list else 0.5
    
    print(f"✓ Calibration complete:")
    print(f"  Yaw: {baseline_yaw:.2f}°, Pitch: {baseline_pitch:.2f}°")
    print(f"  Gaze: ({baseline_gaze_x:.3f}, {baseline_gaze_y:.3f})\n")
    
    cv2.destroyWindow("Calibrating...")
    return baseline_yaw, baseline_pitch, baseline_gaze_x, baseline_gaze_y

# ----------------- Main -----------------
def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("❌ Camera not available")
    
    print("\n" + "="*60)
    print("🎓 EXAM PROCTORING SYSTEM (OpenCV Version)")
    print("="*60)
    print("Initializing systems...")
    print("="*60 + "\n")
    
    sound_detector = SoundDetector()
    face_recognizer = OpenCVFaceRecognizer()
    data_logger = DataLogger(CSV_FILENAME)
    metrics = PerformanceMetrics()
    
    sound_detector.start()
    time.sleep(1)
    sound_detector.auto_calibrate()
    
    register_initial_face(cap, face_recognizer)
    baseline_yaw, baseline_pitch, baseline_gaze_x, baseline_gaze_y = calibrate(cap)
    
    # Buffers
    gaze_x_buf = deque(maxlen=SMOOTH_WINDOW)
    gaze_y_buf = deque(maxlen=SMOOTH_WINDOW)
    yaw_buf = deque(maxlen=SMOOTH_WINDOW)
    pitch_buf = deque(maxlen=SMOOTH_WINDOW)
    
    # State
    eye_closed_counter = 0
    attentive_hold = 0
    inattentive_hold = 0
    status = "UNKNOWN"
    status_color = (0,0,255)
    session_start = time.time()
    frame_count = 0
    
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
            h, w = frame.shape[:2]
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = face_mesh.process(rgb)
            
            frame_count += 1
            session_time = time.time() - session_start

            sound_details = sound_detector.get_detailed_status()
            sound_detected = sound_details['is_detected']
            sound_level = sound_details['rms_level']
            sensitivity = sound_details['sensitivity_percent']
            
            metrics.update_sound(sound_detected)

            if face_recognizer.reference_face is not None:
                verify_result, verify_msg = face_recognizer.verify_face(frame)
                if verify_result is not None:
                    metrics.update_identity(verify_result)
                    print(f"Identity check: {verify_msg}")

            identity_status, identity_color, match_conf = face_recognizer.get_status()

            head_yaw = head_pitch = 0.0
            avg_gx = avg_gy = 0.5
            avg_ear = 0.3
            candidate_status = "NO FACE"
            candidate_color = (128,128,128)

            if res.multi_face_landmarks:
                lm = res.multi_face_landmarks[0].landmark

                yaw, pitch = solve_head_pose(lm, w, h)
                if yaw: 
                    yaw_buf.append(yaw)
                    head_yaw = float(np.mean(yaw_buf))
                if pitch:
                    pitch_buf.append(pitch)
                    head_pitch = float(np.mean(pitch_buf))

                left_iris, right_iris = iris_centers(lm, w, h)
                
                if left_iris is not None:
                    left_eye_left = np.array([lm[LEFT_EYE_CORNERS[0]].x * w, lm[LEFT_EYE_CORNERS[0]].y * h])
                    left_eye_right = np.array([lm[LEFT_EYE_CORNERS[1]].x * w, lm[LEFT_EYE_CORNERS[1]].y * h])
                    right_eye_left = np.array([lm[RIGHT_EYE_CORNERS[0]].x * w, lm[RIGHT_EYE_CORNERS[0]].y * h])
                    right_eye_right = np.array([lm[RIGHT_EYE_CORNERS[1]].x * w, lm[RIGHT_EYE_CORNERS[1]].y * h])
                    left_top = np.array([lm[LEFT_EYE_TOP_BOTTOM[0]].x * w, lm[LEFT_EYE_TOP_BOTTOM[0]].y * h])
                    left_bottom = np.array([lm[LEFT_EYE_TOP_BOTTOM[1]].x * w, lm[LEFT_EYE_TOP_BOTTOM[1]].y * h])
                    right_top = np.array([lm[RIGHT_EYE_TOP_BOTTOM[0]].x * w, lm[RIGHT_EYE_TOP_BOTTOM[0]].y * h])
                    right_bottom = np.array([lm[RIGHT_EYE_TOP_BOTTOM[1]].x * w, lm[RIGHT_EYE_TOP_BOTTOM[1]].y * h])

                    left_gx = (left_iris[0] - left_eye_left[0]) / (left_eye_right[0] - left_eye_left[0] + 1e-6)
                    right_gx = (right_iris[0] - right_eye_left[0]) / (right_eye_right[0] - right_eye_left[0] + 1e-6)
                    left_gy = (left_iris[1] - left_top[1]) / (left_bottom[1] - left_top[1] + 1e-6)
                    right_gy = (right_iris[1] - right_top[1]) / (right_bottom[1] - right_top[1] + 1e-6)
                    avg_gx = (left_gx + right_gx) / 2.0
                    avg_gy = (left_gy + right_gy) / 2.0

                    gaze_x_buf.append(avg_gx)
                    gaze_y_buf.append(avg_gy)
                    smooth_gx = float(np.mean(gaze_x_buf))
                    smooth_gy = float(np.mean(gaze_y_buf))

                    yaw_delta = head_yaw - baseline_yaw
                    pitch_delta = head_pitch - baseline_pitch
                    iris_x_delta = smooth_gx - baseline_gaze_x
                    iris_y_delta = smooth_gy - baseline_gaze_y

                    gaze_dev = abs(iris_x_delta)
                    gaze_dev_y = abs(iris_y_delta)
                    head_yaw_dev = abs(yaw_delta)
                    head_pitch_dev = abs(pitch_delta)
                    
                    left_ear = eye_aspect_ratio(lm, LEFT_EYE_EAR_IDX, w, h)
                    right_ear = eye_aspect_ratio(lm, RIGHT_EYE_EAR_IDX, w, h)
                    avg_ear = (left_ear + right_ear) / 2.0
                    
                    if avg_ear < EAR_THRESH:
                        eye_closed_counter += 1
                    else:
                        eye_closed_counter = max(eye_closed_counter - 1, 0)
                    
                    eyes_closed = eye_closed_counter >= EYE_CLOSED_CONSEC
                    
                    head_turned = (head_yaw_dev > HEAD_YAW_TOL_DEG) or (head_pitch_dev > HEAD_PITCH_TOL_DEG)
                    head_turned_yaw = head_yaw_dev > HEAD_YAW_TOL_DEG
                    head_turned_pitch = head_pitch_dev > HEAD_PITCH_TOL_DEG
                    
                    yaw_compensating = (yaw_delta * iris_x_delta) < 0 and abs(iris_x_delta) > COMPENSATION_MIN
                    pitch_compensating = (pitch_delta * iris_y_delta) < 0 and abs(iris_y_delta) > COMPENSATION_PITCH_MIN
                    any_compensation = yaw_compensating or pitch_compensating
                    
                    total_gaze_deviation = max(gaze_dev, gaze_dev_y)
                    eyes_too_far = total_gaze_deviation > MAX_GAZE_DEVIATION
                    
                    # Decision logic
                    if eyes_closed:
                        candidate_status = "INATTENTIVE (Eyes closed)"
                        candidate_color = (0, 128, 255)
                    elif eyes_too_far:
                        if head_turned:
                            candidate_status = "INATTENTIVE (Head and eyes away)"
                            candidate_color = (0,0,255)
                        else:
                            candidate_status = "INATTENTIVE (Eyes looking away)"
                            candidate_color = (0,0,255)
                    else:
                        if (gaze_dev <= GAZE_THRESH and gaze_dev_y <= GAZE_THRESH):
                            candidate_status = "ATTENTIVE (Direct gaze)"
                            candidate_color = (0,255,0)
                        elif (gaze_dev <= GAZE_RELAXED and gaze_dev_y <= GAZE_RELAXED):
                            if head_turned:
                                if any_compensation:
                                    if (yaw_compensating and abs(iris_x_delta) > COMPENSATION_MIN * 1.5) or \
                                       (pitch_compensating and abs(iris_y_delta) > COMPENSATION_PITCH_MIN * 1.2):
                                        candidate_status = "ATTENTIVE (Compensating)"
                                        candidate_color = (0,255,0)
                                    else:
                                        candidate_status = "ATTENTIVE (Eyes on screen)"
                                        candidate_color = (0,220,0)
                                else:
                                    candidate_status = "ATTENTIVE (Eyes on screen)"
                                    candidate_color = (0,220,0)
                            else:
                                candidate_status = "ATTENTIVE (Normal)"
                                candidate_color = (0,255,0)
                        elif any_compensation and not eyes_too_far:
                            if gaze_dev <= GAZE_RELAXED * 1.3 and gaze_dev_y <= GAZE_RELAXED * 1.3:
                                if (yaw_compensating and abs(iris_x_delta) > COMPENSATION_MIN * 2.5) or \
                                   (pitch_compensating and abs(iris_y_delta) > COMPENSATION_PITCH_MIN * 2):
                                    candidate_status = "ATTENTIVE (Strong compensation)"
                                    candidate_color = (0,200,0)
                                else:
                                    candidate_status = "INATTENTIVE (Insufficient compensation)"
                                    candidate_color = (0,0,255)
                            else:
                                candidate_status = "INATTENTIVE (Eyes deviating)"
                                candidate_color = (0,0,255)
                        elif head_turned_pitch and not head_turned_yaw:
                            if pitch_compensating and gaze_dev_y <= GAZE_RELAXED:
                                candidate_status = "ATTENTIVE (Vertical adjustment)"
                                candidate_color = (0,220,0)
                            else:
                                candidate_status = "INATTENTIVE (Looking up/down)"
                                candidate_color = (0,0,255)
                        else:
                            if head_turned:
                                candidate_status = "INATTENTIVE (Head and eyes away)"
                                candidate_color = (0,0,255)
                            else:
                                candidate_status = "INATTENTIVE (Looking away)"
                                candidate_color = (0,0,255)

                    if candidate_status.startswith("ATTENTIVE"):
                        attentive_hold += 1
                        inattentive_hold = 0
                    else:
                        inattentive_hold += 1
                        attentive_hold = 0

                    if attentive_hold >= HOLD_FRAMES:
                        status = "ATTENTIVE"
                        status_color = (0,255,0)
                    elif inattentive_hold >= HOLD_FRAMES:
                        status = "INATTENTIVE"
                        status_color = (0,0,255)

                    # Display
                    cv2.putText(frame, f"Yaw:{head_yaw:.1f} Pitch:{head_pitch:.1f}", (10,30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200,200,200), 2)
                    
                    gaze_color = (0,255,0) if total_gaze_deviation <= GAZE_RELAXED else (0,165,255) if total_gaze_deviation <= MAX_GAZE_DEVIATION else (0,0,255)
                    cv2.putText(frame, f"Gaze X:{gaze_dev:.3f} Y:{gaze_dev_y:.3f} Max:{total_gaze_deviation:.3f}", (10,55), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, gaze_color, 1)
                    
                    comp_text = ""
                    if yaw_compensating:
                        comp_text = f"YAW-COMP({abs(iris_x_delta):.3f})"
                    if pitch_compensating:
                        comp_text += f" PITCH-COMP({abs(iris_y_delta):.3f})" if comp_text else f"PITCH-COMP({abs(iris_y_delta):.3f})"
                    if comp_text:
                        cv2.putText(frame, comp_text, (10,100), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0,255,255), 1)
                    
                    if eyes_too_far:
                        cv2.putText(frame, "⚠ EYES TOO FAR FROM SCREEN", (10,120), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 2)
                    
                    cv2.putText(frame, f"Status: {candidate_status}", (10,80), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, candidate_color, 2)
                    
                    try:
                        cv2.circle(frame, tuple(left_iris.astype(int)), 3, (0,255,255), -1)
                        cv2.circle(frame, tuple(right_iris.astype(int)), 3, (0,255,255), -1)
                    except:
                        pass
            else:
                inattentive_hold += 1
                attentive_hold = 0
                if inattentive_hold >= HOLD_FRAMES:
                    status = "INATTENTIVE"
                    status_color = (0,0,255)
                cv2.putText(frame, "NO FACE DETECTED", (10,30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,255), 2)

            # Update metrics
            predicted_attentive = status == "ATTENTIVE"
            actual_attentive = candidate_status.startswith("ATTENTIVE")
            metrics.update(predicted_attentive, actual_attentive)
            
            # Print metrics every 30 seconds
            metrics.print_metrics()

            # Log data
            data_logger.log_data({
                'timestamp': datetime.now().isoformat(),
                'session_time': round(session_time, 2),
                'attention_state': status,
                'identity_status': identity_status,
                'sound_detected': sound_detected,
                'sound_level': round(sound_level, 4),
                'head_yaw': round(head_yaw, 2),
                'head_pitch': round(head_pitch, 2),
                'gaze_x': round(avg_gx, 3),
                'gaze_y': round(avg_gy, 3),
                'ear': round(avg_ear, 3),
                'confidence': round(match_conf, 2)
            })

            # Display boxes
            box_height = 70
            box_y = h - box_height - 10
            
            # Attention
            cv2.rectangle(frame, (5, box_y), (250, h-10), (0, 0, 0), -1)
            cv2.rectangle(frame, (5, box_y), (250, h-10), status_color, 2)
            cv2.putText(frame, "ATTENTION", (15, box_y+25), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)
            cv2.putText(frame, status, (15, box_y+50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
            
            # Identity
            mid_x = w//2 - 125
            cv2.rectangle(frame, (mid_x, box_y), (mid_x+250, h-10), (0, 0, 0), -1)
            cv2.rectangle(frame, (mid_x, box_y), (mid_x+250, h-10), identity_color, 2)
            cv2.putText(frame, "IDENTITY", (mid_x+10, box_y+25), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, identity_color, 2)
            cv2.putText(frame, identity_status, (mid_x+10, box_y+50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, identity_color, 2)
            
            # Sound - RED when detected
            sound_color = (0, 0, 255) if sound_detected else (100, 100, 100)
            sound_text = f"SPEAKING ({sensitivity:.0f}%)" if sound_detected else f"QUIET ({sensitivity:.0f}%)"
            
            cv2.rectangle(frame, (w-255, box_y), (w-5, h-10), (0, 0, 0), -1)
            cv2.rectangle(frame, (w-255, box_y), (w-5, h-10), sound_color, 2)
            cv2.putText(frame, "SOUND", (w-240, box_y+25), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, sound_color, 2)
            cv2.putText(frame, sound_text, (w-240, box_y+50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, sound_color, 1)
            
            # Sound bar
            bar_w = 200
            bar_h = 8
            bar_x = w - 245
            bar_y_pos = box_y + 55
            filled = int(bar_w * min(sensitivity / 100, 1.0))
            
            cv2.rectangle(frame, (bar_x, bar_y_pos), (bar_x + bar_w, bar_y_pos + bar_h), 
                         (50, 50, 50), -1)
            if filled > 0:
                cv2.rectangle(frame, (bar_x, bar_y_pos), (bar_x + filled, bar_y_pos + bar_h), 
                             sound_color, -1)

            # Reference face thumbnail
            if face_recognizer.reference_image is not None:
                thumb_size = 100
                thumb = cv2.resize(face_recognizer.reference_image, (thumb_size, thumb_size))
                frame[10:10+thumb_size, w-thumb_size-10:w-10] = thumb
                cv2.rectangle(frame, (w-thumb_size-10, 10), (w-10, 10+thumb_size), (0, 255, 0), 2)
                cv2.putText(frame, "Reference", (w-thumb_size-10, thumb_size+25), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
            
            # Session info
            mins = int(session_time // 60)
            secs = int(session_time % 60)
            cv2.putText(frame, f"Session: {mins:02d}:{secs:02d}", (10, h-15), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            cv2.putText(frame, f"Frames: {frame_count}", (150, h-15), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

            cv2.imshow("Exam Proctoring System", frame)
            
            # Controls
            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC
                break
            elif key == ord('r'):
                print("\n🔄 Recalibrating...")
                baseline_yaw, baseline_pitch, baseline_gaze_x, baseline_gaze_y = calibrate(cap)
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
        print("\n⚠️ Interrupted")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("\n" + "="*60)
        print("🛑 SHUTTING DOWN")
        print("="*60)
        
        sound_detector.stop()
        data_logger.finalize()
        cap.release()
        cv2.destroyAllWindows()
        
        # Print final comprehensive report
        print("\n" + metrics.get_summary_report())
        
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