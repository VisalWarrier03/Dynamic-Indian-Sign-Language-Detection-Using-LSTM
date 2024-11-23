import os
import numpy as np
import tensorflow as tf
import mediapipe as mp
import cv2
import pandas as pd

# Constants (match the training script)
MAX_SEQUENCE_LENGTH = 150  
FEATURE_DIM = 225  # Pose (33), left hand (21), right hand (21), all with x, y, z

class SignLanguageRecognizer:
    def __init__(self, model_path, csv_path):
        # MediaPipe Holistic
        self.mp_holistic = mp.solutions.holistic
        self.holistic = self.mp_holistic.Holistic(
            static_image_mode=False, 
            min_detection_confidence=0.5, 
            min_tracking_confidence=0.5
        )
        
        # Load the trained model
        self.model = tf.keras.models.load_model(model_path)
        
        # Load class names
        df = pd.read_csv(csv_path)
        class_counts = df['Adjectives'].value_counts()
        self.classes = class_counts[class_counts >= 18].index.tolist()
        
        # Inference variables
        self.current_sequence = []
        self.predictions = []
        self.prediction_history = []
        
        # Configuration parameters
        self.confidence_threshold = 0.5  # More conservative threshold
        self.stability_threshold = 3  # Frames for stable prediction
        self.debug_mode = True  # Enable detailed logging

    def normalize_landmarks(self, landmarks):
        """
        Normalize landmarks relative to their center and scale.
        
        Args:
            landmarks (np.array): Input landmarks with shape (n, 3)
        
        Returns:
            np.array: Normalized landmarks
        """
        if len(landmarks) == 0:
            return np.zeros((landmarks.shape[0], 3))
        
        # Compute center
        center = np.mean(landmarks, axis=0)
        
        # Normalize by subtracting center and scaling
        normalized = (landmarks - center)
        
        # Add small epsilon to avoid division by zero
        scale = np.std(normalized, axis=0) + 1e-8
        normalized /= scale
        
        return normalized

    def extract_frame_features(self, frame):
        """Extract skeletal key points from a single frame and scale to 1920x1080 resolution."""
        # Dimensions for scaling
        target_width = 1920
        target_height = 1080
        frame_height, frame_width, _ = frame.shape

        # Convert frame to RGB for MediaPipe
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process frame with MediaPipe
        results = self.holistic.process(frame_rgb)

        # Scaling factors
        scale_x = target_width / frame_width
        scale_y = target_height / frame_height

        # Extract landmarks and scale coordinates
        pose = (
            np.array([[lm.x * frame_width * scale_x, lm.y * frame_height * scale_y, lm.z] 
                    for lm in results.pose_landmarks.landmark])
            if results.pose_landmarks else np.zeros((33, 3))
        )
        left_hand = (
            np.array([[lm.x * frame_width * scale_x, lm.y * frame_height * scale_y, lm.z] 
                    for lm in results.left_hand_landmarks.landmark])
            if results.left_hand_landmarks else np.zeros((21, 3))
        )
        right_hand = (
            np.array([[lm.x * frame_width * scale_x, lm.y * frame_height * scale_y, lm.z] 
                    for lm in results.right_hand_landmarks.landmark])
            if results.right_hand_landmarks else np.zeros((21, 3))
        )

        # Normalize landmarks
        pose = self.normalize_landmarks(pose)
        left_hand = self.normalize_landmarks(left_hand)
        right_hand = self.normalize_landmarks(right_hand)

        # Ensure correct dimensions
        pose = pose[:33] if len(pose) > 33 else np.pad(pose, ((0, max(0, 33-len(pose))), (0, 0)))
        left_hand = left_hand[:21] if len(left_hand) > 21 else np.pad(left_hand, ((0, max(0, 21-len(left_hand))), (0, 0)))
        right_hand = right_hand[:21] if len(right_hand) > 21 else np.pad(right_hand, ((0, max(0, 21-len(right_hand))), (0, 0)))

        # Combine all landmarks into a single frame feature vector
        frame_features = np.concatenate([pose.flatten(), left_hand.flatten(), right_hand.flatten()])
        return frame_features

    def predict_sign(self, sequence):
        """
        Predict top 5 signs from a sequence of frames.
        
        Args:
            sequence (np.array): Sequence of frame features
        
        Returns:
            list: Top 5 predictions with class and confidence
        """
        # Ensure sequence is the right length
        if sequence.shape[0] < MAX_SEQUENCE_LENGTH:
            padding = np.zeros((MAX_SEQUENCE_LENGTH - sequence.shape[0], FEATURE_DIM))
            sequence = np.vstack([sequence, padding])
        elif sequence.shape[0] > MAX_SEQUENCE_LENGTH:
            sequence = sequence[:MAX_SEQUENCE_LENGTH]
        
        # Reshape for model input and predict
        sequence = sequence.reshape(1, MAX_SEQUENCE_LENGTH, FEATURE_DIM)
        predictions = self.model.predict(sequence)[0]
        
        # Get top 5 predictions
        top_5_indices = predictions.argsort()[-5:][::-1]
        top_5_predictions = [(self.classes[idx], predictions[idx]) for idx in top_5_indices]
        
        # Debug logging
        if self.debug_mode:
            print("Raw Predictions:")
            for cls, conf in top_5_predictions:
                print(f"{cls}: {conf:.4f}")
        
        return top_5_predictions

    def run_inference(self, source=0, display=True):
        """
        Run real-time sign language recognition.
        
        Args:
            source (int/str): Video source (webcam or video file)
            display (bool): Whether to show video feed with predictions
        """
        cap = cv2.VideoCapture(source)
        
        # Reset sequence and prediction history
        self.current_sequence = []
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Extract features from current frame
            frame_features = self.extract_frame_features(frame)
            
            # Add to sequence
            self.current_sequence.append(frame_features)
            
            # Limit sequence length
            if len(self.current_sequence) > MAX_SEQUENCE_LENGTH:
                self.current_sequence = self.current_sequence[-MAX_SEQUENCE_LENGTH:]
            
            # Predict when enough frames are collected
            if len(self.current_sequence) >= 30:  # Minimum frames before inference
                sequence_array = np.array(self.current_sequence)
                top_5_predictions = self.predict_sign(sequence_array)
                
                # Display results if enabled
                if display:
                    stable_pred = top_5_predictions[0]
                    cls, conf = stable_pred
                    display_text = f"{cls}: {conf:.2f}"
                    cv2.putText(frame, display_text, 
                                (10, 50), 
                                cv2.FONT_HERSHEY_SIMPLEX, 
                                1, 
                                (0, 255, 0), 
                                2)
                
                cv2.imshow('Sign Language Recognition', frame)
            
            # Exit on 'q' key
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        # Cleanup
        cap.release()
        cv2.destroyAllWindows()

# Main execution
if __name__ == "__main__":
    MODEL_PATH = r"adjectives_lstm_model_16.keras"
    CSV_PATH = "Adjectives_videos.csv"
    VIDEO_PATH = r"test_video.mp4"
    
    recognizer = SignLanguageRecognizer(MODEL_PATH, CSV_PATH)
    recognizer.run_inference(source=VIDEO_PATH) # 0 for default webcam

