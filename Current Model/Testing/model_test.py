import tensorflow as tf
import numpy as np
import cv2
import mediapipe as mp
import os
from datetime import datetime
import pandas as pd

# Initialize MediaPipe
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles


# [Previous imports and constants remain the same]
MAX_SEQUENCE_LENGTH = 150
FEATURE_DIM = 225
DESKTOP_PATH = os.path.expanduser("~/Desktop")
BASE_MODEL_DIR = r"Final_Run_ver3"
CLASSES = ['cold', 'cool', 'dry', 'happy', 'healthy', 'hot', 'loud', 'new', 'quiet', 'sick', 'warm']

class SignLanguagePredictor:
    # [Previous SignLanguagePredictor class implementation remains the same]
    def __init__(self):
        self.models = self._load_all_models()
        self.holistic = mp_holistic.Holistic(
            static_image_mode=False,
            model_complexity=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

    def _load_all_models(self):
        models = {}
        for model_name in os.listdir(BASE_MODEL_DIR):
            model_path = os.path.join(BASE_MODEL_DIR, model_name, 'saved_model')
            if os.path.exists(model_path):
                try:
                    model = tf.saved_model.load(model_path)
                    models[model_name] = model
                    print(f"Loaded model: {model_name}")
                except Exception as e:
                    print(f"Error loading model {model_name}: {str(e)}")
        return models

    def extract_features(self, frame):
        results = self.holistic.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        
        pose = np.array([[lm.x, lm.y, lm.z] for lm in results.pose_landmarks.landmark]) if results.pose_landmarks else np.zeros((33, 3))
        left_hand = np.array([[lm.x, lm.y, lm.z] for lm in results.left_hand_landmarks.landmark]) if results.left_hand_landmarks else np.zeros((21, 3))
        right_hand = np.array([[lm.x, lm.y, lm.z] for lm in results.right_hand_landmarks.landmark]) if results.right_hand_landmarks else np.zeros((21, 3))
        
        if pose.shape != (33, 3): pose = np.zeros((33, 3))
        if left_hand.shape != (21, 3): left_hand = np.zeros((21, 3))
        if right_hand.shape != (21, 3): right_hand = np.zeros((21, 3))
        
        features = np.concatenate([pose.flatten(), left_hand.flatten(), right_hand.flatten()])
        return features

def process_video(video_path):
    """Process video and return predictions in a pandas DataFrame."""
    predictor = SignLanguagePredictor()
    cap = cv2.VideoCapture(video_path)
    sequence_buffer = []
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        features = predictor.extract_features(frame)
        sequence_buffer.append(features)
        
        if len(sequence_buffer) >= MAX_SEQUENCE_LENGTH:
            break
    
    cap.release()
    
    if len(sequence_buffer) < MAX_SEQUENCE_LENGTH:
        padding = np.zeros((MAX_SEQUENCE_LENGTH - len(sequence_buffer), FEATURE_DIM))
        sequence_buffer = np.vstack([sequence_buffer, padding])
    else:
        sequence_buffer = sequence_buffer[:MAX_SEQUENCE_LENGTH]
    
    sequence = np.expand_dims(np.array(sequence_buffer), axis=0)
    
    # Store results
    results = []
    
    for model_name, model in predictor.models.items():
        try:
            pred = model.signatures['serving_default'](tf.constant(sequence.astype(np.float32)))
            pred_array = next(iter(pred.values())).numpy()[0]
            
            # Get top 3 predictions
            top_3_indices = np.argsort(pred_array)[-3:][::-1]
            top_3_classes = [CLASSES[i] for i in top_3_indices]
            top_3_confidences = pred_array[top_3_indices]
            
            results.append({
                'Model': model_name,
                'Predicted_Class': top_3_classes[0],
                'Confidence': top_3_confidences[0],
                'Second_Guess': top_3_classes[1],
                'Second_Confidence': top_3_confidences[1],
                'Third_Guess': top_3_classes[2],
                'Third_Confidence': top_3_confidences[2]
            })
            
        except Exception as e:
            print(f"Error with model {model_name}: {str(e)}")
            continue
    
    return pd.DataFrame(results)

def main():
    video_path = input("Enter the path to the video file: ")
    print("Processing video...")
    
    # Get predictions
    df = process_video(video_path)
    
    if df.empty:
        print("No predictions were generated!")
        return
    
    # Format confidences as percentages
    confidence_cols = ['Confidence', 'Second_Confidence', 'Third_Confidence']
    df[confidence_cols] = df[confidence_cols].apply(lambda x: x * 100)
    
    # Display formatted table
    print("\nPrediction Results:")
    print("=" * 100)
    print(df.to_string(index=False, float_format=lambda x: '{:.2f}'.format(x)))
    
    # Save to CSV
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = os.path.join(DESKTOP_PATH, f"sign_language_predictions_{timestamp}.csv")
    df.to_csv(csv_path, index=False)
    print(f"\nResults saved to: {csv_path}")

if __name__ == "__main__":
    main()