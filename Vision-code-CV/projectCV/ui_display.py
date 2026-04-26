"""
UI Display Module for Exam Proctoring System
"""
import cv2
import time


class UIDisplay:
    def __init__(self):
        self.session_start = time.time()
        self.frame_count = 0
        
    def draw_status_boxes(self, frame, attention_result, identity_status, identity_color, sound_details):
        """Draw status indicator boxes at bottom of frame"""
        h, w = frame.shape[:2]
        box_height = 70
        box_y = h - box_height - 10
        
        # Attention Status Box
        status = attention_result['status']
        status_color = attention_result['color']
        
        cv2.rectangle(frame, (5, box_y), (250, h-10), (0, 0, 0), -1)
        cv2.rectangle(frame, (5, box_y), (250, h-10), status_color, 2)
        cv2.putText(frame, "ATTENTION", (15, box_y+25), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)
        cv2.putText(frame, status, (15, box_y+50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
        
        # Identity Status Box
        mid_x = w//2 - 125
        cv2.rectangle(frame, (mid_x, box_y), (mid_x+250, h-10), (0, 0, 0), -1)
        cv2.rectangle(frame, (mid_x, box_y), (mid_x+250, h-10), identity_color, 2)
        cv2.putText(frame, "IDENTITY", (mid_x+10, box_y+25), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, identity_color, 2)
        cv2.putText(frame, identity_status, (mid_x+10, box_y+50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, identity_color, 2)
        
        # Sound Status Box (RED when detected)
        sound_detected = sound_details['is_detected']
        sensitivity = sound_details['sensitivity_percent']
        
        sound_color = (0, 0, 255) if sound_detected else (100, 100, 100)
        sound_text = f"SPEAKING ({sensitivity:.0f}%)" if sound_detected else f"QUIET ({sensitivity:.0f}%)"
        
        cv2.rectangle(frame, (w-255, box_y), (w-5, h-10), (0, 0, 0), -1)
        cv2.rectangle(frame, (w-255, box_y), (w-5, h-10), sound_color, 2)
        cv2.putText(frame, "SOUND", (w-240, box_y+25), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, sound_color, 2)
        cv2.putText(frame, sound_text, (w-240, box_y+50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, sound_color, 1)
        
        # Sound level bar
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
    
    def draw_tracking_info(self, frame, attention_result, tracking_data):
        """Draw head pose and gaze information"""
        if tracking_data is None or attention_result['details'] is None:
            cv2.putText(frame, "NO FACE DETECTED", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
            return
        
        details = attention_result['details']
        candidate_status = attention_result['candidate_status']
        candidate_color = attention_result['candidate_color']
        
        # Head pose
        cv2.putText(frame, f"Yaw:{details['head_yaw']:.1f} Pitch:{details['head_pitch']:.1f}", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
        
        # Gaze deviation with color coding
        total_dev = details['total_gaze_deviation']
        gaze_color = (0, 255, 0) if total_dev <= 0.12 else (0, 165, 255) if total_dev <= 0.20 else (0, 0, 255)
        
        cv2.putText(frame, f"Gaze X:{details['gaze_dev']:.3f} Y:{details['gaze_dev_y']:.3f} Max:{total_dev:.3f}", 
                   (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.5, gaze_color, 1)
        
        # Compensation indicators
        comp_text = ""
        if details['yaw_compensating']:
            comp_text = f"YAW-COMP({abs(details['iris_x_delta']):.3f})"
        if details['pitch_compensating']:
            comp_text += f" PITCH-COMP({abs(details['iris_y_delta']):.3f})" if comp_text else f"PITCH-COMP({abs(details['iris_y_delta']):.3f})"
        if comp_text:
            cv2.putText(frame, comp_text, (10, 100), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
        
        # Warning if eyes too far
        if details['eyes_too_far']:
            cv2.putText(frame, "⚠ EYES TOO FAR FROM SCREEN", (10, 120), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        
        # Candidate status
        cv2.putText(frame, f"Status: {candidate_status}", (10, 80), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, candidate_color, 2)
        
        # Draw iris positions
        if tracking_data.get('left_iris') is not None:
            try:
                cv2.circle(frame, tuple(tracking_data['left_iris'].astype(int)), 3, (0, 255, 255), -1)
                cv2.circle(frame, tuple(tracking_data['right_iris'].astype(int)), 3, (0, 255, 255), -1)
            except:
                pass
    
    def draw_reference_thumbnail(self, frame, reference_image):
        """Draw reference face thumbnail"""
        if reference_image is None:
            return
        
        h, w = frame.shape[:2]
        thumb_size = 100
        thumb = cv2.resize(reference_image, (thumb_size, thumb_size))
        frame[10:10+thumb_size, w-thumb_size-10:w-10] = thumb
        cv2.rectangle(frame, (w-thumb_size-10, 10), (w-10, 10+thumb_size), (0, 255, 0), 2)
        cv2.putText(frame, "Reference", (w-thumb_size-10, thumb_size+25), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
    
    def draw_session_info(self, frame):
        """Draw session time and frame count"""
        h, w = frame.shape[:2]
        session_time = time.time() - self.session_start
        mins = int(session_time // 60)
        secs = int(session_time % 60)
        
        cv2.putText(frame, f"Session: {mins:02d}:{secs:02d}", (10, h-15), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        cv2.putText(frame, f"Frames: {self.frame_count}", (150, h-15), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    
    def render_frame(self, frame, attention_result, tracking_data, identity_status, 
                     identity_color, sound_details, reference_image):
        """Main rendering function - combines all UI elements"""
        self.frame_count += 1
        
        self.draw_tracking_info(frame, attention_result, tracking_data)
        self.draw_status_boxes(frame, attention_result, identity_status, identity_color, sound_details)
        self.draw_reference_thumbnail(frame, reference_image)
        self.draw_session_info(frame)
        
        return frame
    
    def get_session_time(self):
        """Get current session time in seconds"""
        return time.time() - self.session_start
    
    def get_frame_count(self):
        """Get total frame count"""
        return self.frame_count