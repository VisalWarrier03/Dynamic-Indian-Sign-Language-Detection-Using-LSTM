import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import cv2
import numpy as np
import pandas as pd
import mediapipe as mp
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

# Paths
csv_path = "Adjectives_videos.csv"
model_save_path = "adjectives_lstm_model_with_interpolation.keras"

# Initialize MediaPipe
mp_holistic = mp.solutions.holistic

# Parameters
sequence_length = 100
pose_feature_dim = 33 * 4
hand_feature_dim = 21 * 3
features_per_frame = pose_feature_dim + (2 * hand_feature_dim)
batch_size = 16
epochs = 50
input_width, input_height = 960, 540

# Load CSV and validate
data_df = pd.read_csv(csv_path)
if 'Adjectives' not in data_df.columns or 'Path' not in data_df.columns:
    raise ValueError("CSV file must contain 'Adjectives' and 'Path' columns")

# Filter classes with fewer than 18 videos
class_counts = data_df['Adjectives'].value_counts()
valid_classes = class_counts[class_counts >= 18].index
data_df = data_df[data_df['Adjectives'].isin(valid_classes)]

# Class mapping
class_labels = data_df['Adjectives'].unique().tolist()
num_classes = len(class_labels)
class_map = {label: index for index, label in enumerate(class_labels)}

# Feature extraction with interpolation
def extract_keypoints_with_interpolation(frames_keypoints, sequence_length):
    num_frames = len(frames_keypoints)
    if num_frames < sequence_length:
        # Interpolate frames to match the desired sequence length
        x = np.linspace(0, num_frames - 1, num_frames)
        x_new = np.linspace(0, num_frames - 1, sequence_length)
        interpolated_frames = np.array([interp1d(x, np.array(frames_keypoints)[:, i], kind='linear', fill_value="extrapolate")(x_new)
                                        for i in range(features_per_frame)]).T
        return interpolated_frames
    else:
        indices = np.linspace(0, num_frames - 1, sequence_length).astype(int)
        return np.array([frames_keypoints[i] for i in indices], dtype=np.float32)

# Extract keypoints function
def extract_keypoints(results):
    pose = np.zeros(pose_feature_dim, dtype=np.float32)
    left_hand = np.zeros(hand_feature_dim, dtype=np.float32)
    right_hand = np.zeros(hand_feature_dim, dtype=np.float32)

    if results.pose_landmarks:
        pose_landmarks = np.array([[lm.x, lm.y, lm.z, lm.visibility] for lm in results.pose_landmarks.landmark]).flatten()
        pose[:len(pose_landmarks)] = pose_landmarks

    if results.left_hand_landmarks:
        left_hand_landmarks = np.array([[lm.x, lm.y, lm.z] for lm in results.left_hand_landmarks.landmark]).flatten()
        left_hand[:len(left_hand_landmarks)] = left_hand_landmarks

    if results.right_hand_landmarks:
        right_hand_landmarks = np.array([[lm.x, lm.y, lm.z] for lm in results.right_hand_landmarks.landmark]).flatten()
        right_hand[:len(right_hand_landmarks)] = right_hand_landmarks

    return np.concatenate([pose, left_hand, right_hand])

# Process video with interpolation
def process_video(video_path):
    cap = cv2.VideoCapture(video_path)
    frames_keypoints = []

    with mp_holistic.Holistic(min_detection_confidence=0.7, min_tracking_confidence=0.7) as holistic:
        while cap.isOpened() and len(frames_keypoints) < sequence_length:
            ret, frame = cap.read()
            if not ret:
                break
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = holistic.process(frame_rgb)
            keypoints = extract_keypoints(results)
            frames_keypoints.append(keypoints)

    cap.release()
    if len(frames_keypoints) == 0:
        return np.zeros((sequence_length, features_per_frame), dtype=np.float32)
    
    return extract_keypoints_with_interpolation(frames_keypoints, sequence_length)

# LSTM Model
def create_lstm_model(sequence_length, input_dim, num_classes):
    model = Sequential()
    model.add(LSTM(128, return_sequences=True, input_shape=(sequence_length, input_dim)))
    model.add(BatchNormalization())
    model.add(LSTM(128))
    model.add(BatchNormalization())
    model.add(Dropout(0.4))
    model.add(Dense(256, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.4))
    model.add(Dense(num_classes, activation='softmax'))
    model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Process videos
X, y = [], []
for index, row in data_df.iterrows():
    video_path = row['Path']
    label = class_map[row['Adjectives']]
    try:
        keypoints = process_video(video_path)
        X.append(keypoints)
        y.append(label)
    except Exception as e:
        print(f"Failed to process video {video_path}: {str(e)}")

X = np.array(X, dtype=np.float32)
y = to_categorical(y, num_classes=num_classes)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
input_dim = features_per_frame
model = create_lstm_model(sequence_length, input_dim, num_classes)
history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=epochs,
    batch_size=batch_size,
    callbacks=[
        tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True),
        tf.keras.callbacks.ModelCheckpoint(model_save_path, save_best_only=True)
    ]
)

# Evaluate the model
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {test_accuracy:.4f}, Test Loss: {test_loss:.4f}")

# Confusion Matrix
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true_classes = np.argmax(y_test, axis=1)

conf_matrix = confusion_matrix(y_true_classes, y_pred_classes)
plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=class_labels, yticklabels=class_labels)
plt.title('Confusion Matrix')
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.show()

# Classification Report
print("\nClassification Report:")
print(classification_report(y_true_classes, y_pred_classes, target_names=class_labels))

# Model Summary
print("\nModel Summary:")
model.summary()

# Save the model
model.save(model_save_path)

# Plot training history
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.show()
