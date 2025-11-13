import numpy as np
import cv2
from PIL import Image
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.optimizers import Adam
from threading import Thread
import time
import pandas as pd
import os

# Load the Haar cascade for face detection
face_cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
if os.path.exists(face_cascade_path):
    face_cascade = cv2.CascadeClassifier(face_cascade_path)
else:
    face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

# Define the CNN model
def create_emotion_model():
    emotion_model = Sequential()
    
    emotion_model.add(Conv2D(32, kernel_size=(3,3), activation='relu', input_shape=(48,48,1)))
    emotion_model.add(Conv2D(64, kernel_size=(3,3), activation='relu'))
    emotion_model.add(MaxPooling2D(pool_size=(2,2)))
    emotion_model.add(Dropout(0.25))
    
    emotion_model.add(Conv2D(128, kernel_size=(3,3), activation='relu'))
    emotion_model.add(MaxPooling2D(pool_size=(2,2)))
    emotion_model.add(Conv2D(128, kernel_size=(3,3), activation='relu'))
    emotion_model.add(MaxPooling2D(pool_size=(2,2)))
    emotion_model.add(Dropout(0.25))
    
    emotion_model.add(Flatten())
    emotion_model.add(Dense(1024, activation='relu'))
    emotion_model.add(Dropout(0.5))
    emotion_model.add(Dense(7, activation='softmax'))
    
    return emotion_model

# Initialize model
emotion_model = create_emotion_model()

# Try to load pre-trained weights
try:
    emotion_model.load_weights('emotion_model1.h5')
    print("Loaded pre-trained emotion model weights")
except:
    print("Warning: Could not load model.h5. Using untrained model.")

# Set OpenCL usage to false
cv2.ocl.setUseOpenCL(False)

# Emotion dictionaries
emotion_dict = {
    0: "angry", 
    1: "disgusted", 
    2: "fearful", 
    3: "happy", 
    4: "neutral", 
    5: "sad", 
    6: "surprised"
}

# Emotion colors for visualization
emotion_colors = {
    "angry": (0, 0, 255),      # Red
    "disgusted": (128, 0, 128), # Purple
    "fearful": (255, 165, 0),   # Orange
    "happy": (0, 255, 0),       # Green
    "neutral": (255, 255, 255), # White
    "sad": (255, 0, 0),         # Blue
    "surprised": (0, 255, 255)  # Yellow
}

class WebcamVideoStream:
    def __init__(self, src=0, resolution=(640, 480)):
        self.stream = cv2.VideoCapture(src)
        self.stream.set(cv2.CAP_PROP_FRAME_WIDTH, resolution[0])
        self.stream.set(cv2.CAP_PROP_FRAME_HEIGHT, resolution[1])
        self.stream.set(cv2.CAP_PROP_FPS, 30)
        (self.grabbed, self.frame) = self.stream.read()
        self.stopped = False

    def start(self):
        Thread(target=self.update, args=(), daemon=True).start()
        return self

    def update(self):
        while True:
            if self.stopped:
                return
            (self.grabbed, self.frame) = self.stream.read()

    def read(self):
        return self.frame

    def stop(self):
        self.stopped = True
        self.stream.release()

class VideoCamera:
    def __init__(self):
        self.last_recognition_time = 0
        self.recognition_interval = 1.0  # Process emotion every 1 second
        self.current_emotion = "neutral"
        self.current_confidence = 0.0
        self.emotion_history = []
        self.max_history = 10
        self.video_stream = None
        self.no_face_counter = 0
        self.max_no_face = 30  # Reset to neutral after 30 frames with no face

    def start_stream(self):
        if self.video_stream is None:
            self.video_stream = WebcamVideoStream(src=0).start()
            time.sleep(2.0)  # Allow camera to warm up

    def stop_stream(self):
        if self.video_stream:
            self.video_stream.stop()
            self.video_stream = None

    def get_frame(self):
        """
        Returns processed frame and emotion data
        """
        if self.video_stream is None:
            self.start_stream()

        # Read frame from video stream
        frame = self.video_stream.read()
        if frame is None:
            return self._create_error_frame(), self._get_emotion_data()

        # Resize frame for processing
        frame = cv2.resize(frame, (640, 480))
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        faces = face_cascade.detectMultiScale(
            gray, 
            scaleFactor=1.3, 
            minNeighbors=5, 
            minSize=(30, 30),
            flags=cv2.CASCADE_SCALE_IMAGE
        )

        current_time = time.time()
        face_detected = len(faces) > 0

        # Process emotion detection
        if face_detected and current_time - self.last_recognition_time >= self.recognition_interval:
            self._process_emotion_detection(gray, faces[0])  # Use the first detected face
            self.last_recognition_time = current_time
            self.no_face_counter = 0
        elif not face_detected:
            self.no_face_counter += 1
            if self.no_face_counter > self.max_no_face:
                self.current_emotion = "neutral"
                self.current_confidence = 0.0

        # Draw face rectangles and emotion labels
        frame = self._draw_face_annotations(frame, faces)
        
        # Add emotion information overlay
        frame = self._add_info_overlay(frame, face_detected)

        # Encode frame as JPEG
        ret, jpeg = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        
        return jpeg.tobytes(), self._get_emotion_data()

    def _process_emotion_detection(self, gray_frame, face_coords):
        """
        Process emotion detection for a detected face
        """
        x, y, w, h = face_coords
        
        # Extract face region
        roi_gray = gray_frame[y:y + h, x:x + w]
        
        try:
            # Preprocess face for emotion recognition
            roi_gray = cv2.resize(roi_gray, (48, 48))
            roi_gray = roi_gray.astype('float32') / 255.0
            roi_gray = np.expand_dims(roi_gray, axis=0)
            roi_gray = np.expand_dims(roi_gray, axis=-1)

            # Predict emotion
            prediction = emotion_model.predict(roi_gray, verbose=0)
            emotion_index = int(np.argmax(prediction))
            confidence = float(np.max(prediction))

            # Update current emotion with smoothing
            new_emotion = emotion_dict[emotion_index]
            
            # Add to emotion history for smoothing
            self.emotion_history.append({
                'emotion': new_emotion,
                'confidence': confidence,
                'timestamp': time.time()
            })
            
            # Keep only recent history
            if len(self.emotion_history) > self.max_history:
                self.emotion_history.pop(0)
            
            # Smooth emotion detection
            self._smooth_emotion_detection()
            
        except Exception as e:
            print(f"Emotion detection error: {e}")
            self.current_emotion = "neutral"
            self.current_confidence = 0.0

    def _smooth_emotion_detection(self):
        """
        Smooth emotion detection using recent history
        """
        if len(self.emotion_history) < 3:
            # Not enough data for smoothing
            if self.emotion_history:
                latest = self.emotion_history[-1]
                self.current_emotion = latest['emotion']
                self.current_confidence = latest['confidence']
            return

        # Get recent emotions with high confidence
        recent_emotions = [
            item for item in self.emotion_history[-5:] 
            if item['confidence'] > 0.3
        ]

        if not recent_emotions:
            self.current_emotion = "neutral"
            self.current_confidence = 0.0
            return

        # Count emotion occurrences
        emotion_counts = {}
        total_confidence = 0
        
        for item in recent_emotions:
            emotion = item['emotion']
            confidence = item['confidence']
            
            if emotion not in emotion_counts:
                emotion_counts[emotion] = {'count': 0, 'total_confidence': 0}
            
            emotion_counts[emotion]['count'] += 1
            emotion_counts[emotion]['total_confidence'] += confidence
            total_confidence += confidence

        # Find dominant emotion
        dominant_emotion = max(
            emotion_counts.keys(), 
            key=lambda x: emotion_counts[x]['count'] * emotion_counts[x]['total_confidence']
        )
        
        # Calculate average confidence for dominant emotion
        dominant_data = emotion_counts[dominant_emotion]
        avg_confidence = dominant_data['total_confidence'] / dominant_data['count']
        
        # Only update if confidence is reasonable
        if avg_confidence > 0.4 or dominant_data['count'] >= 3:
            self.current_emotion = dominant_emotion
            self.current_confidence = min(avg_confidence, 1.0)

    def _draw_face_annotations(self, frame, faces):
        """
        Draw face rectangles and emotion labels on frame
        """
        for (x, y, w, h) in faces:
            # Get color for current emotion
            color = emotion_colors.get(self.current_emotion, (255, 255, 255))
            
            # Draw face rectangle
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            
            # Draw emotion label with background
            label = f"{self.current_emotion.title()}: {self.current_confidence:.2f}"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
            
            # Draw label background
            cv2.rectangle(
                frame, 
                (x, y - label_size[1] - 10), 
                (x + label_size[0], y), 
                color, 
                -1
            )
            
            # Draw label text
            cv2.putText(
                frame, 
                label, 
                (x, y - 5), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.7, 
                (0, 0, 0), 
                2
            )

        return frame

    def _add_info_overlay(self, frame, face_detected):
        """
        Add information overlay to frame
        """
        # Add status information
        status_color = (0, 255, 0) if face_detected else (0, 0, 255)
        status_text = "Face Detected" if face_detected else "No Face Detected"
        
        cv2.putText(
            frame, 
            status_text, 
            (10, 30), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            0.7, 
            status_color,
            2
        )
        
        # Add current emotion info
        emotion_text = f"Current Emotion: {self.current_emotion.title()}"
        cv2.putText(
            frame, 
            emotion_text, 
            (10, 60), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            0.6, 
            (255, 255, 255), 
            2
        )
        
        # Add confidence info
        confidence_text = f"Confidence: {self.current_confidence:.2f}"
        cv2.putText(
            frame, 
            confidence_text, 
            (10, 90), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            0.6, 
            (255, 255, 255), 
            2
        )

        return frame

    def _create_error_frame(self):
        """
        Create an error frame when camera is not available
        """
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(
            frame, 
            "Camera not available", 
            (200, 240), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            1, 
            (0, 0, 255), 
            2
        )
        ret, jpeg = cv2.imencode('.jpg', frame)
        return jpeg.tobytes()

    def _get_emotion_data(self):
        """
        Return current emotion data
        """
        return {
            'emotion': self.current_emotion,
            'confidence': self.current_confidence,
            'timestamp': time.time()
        }

    def get_emotion_history(self):
        """
        Return recent emotion history
        """
        return self.emotion_history.copy()

    def reset_emotion(self):
        """
        Reset emotion detection
        """
        self.current_emotion = "neutral"
        self.current_confidence = 0.0
        self.emotion_history.clear()
        self.no_face_counter = 0

# Legacy function for backward compatibility
def music_rec(emotion_index=4):
    """
    Legacy function for music recommendation based on emotion
    """
    emotion_name = emotion_dict.get(emotion_index, "neutral")
    
    # Create a simple DataFrame for compatibility
    data = {
        "Artist": [f"Recommended content for {emotion_name} mood"],
        "Emotion": [emotion_name]
    }
    return pd.DataFrame(data)