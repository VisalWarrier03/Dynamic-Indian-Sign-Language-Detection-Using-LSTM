import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import numpy as np
import tensorflow as tf
import mediapipe as mp
import cv2
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Constants
CSV_PATH = "\Adjectives_videos.csv"
MODEL_SAVE_PATH = r"adjectives_stgcn_lstm_model.keras"
NUM_LANDMARKS = 33 + 21 + 21  # Pose + Left Hand + Right Hand

# MediaPipe Holistic
mp_holistic = mp.solutions.holistic

# Helper Functions
def read_video(video_path):
    """Reads frames from a video."""
    cap = cv2.VideoCapture(video_path)
    frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)
    cap.release()
    return frames

def extract_features(video_path):
    """Extracts skeletal features using MediaPipe Holistic."""
    holistic = mp_holistic.Holistic(static_image_mode=False)
    frames = read_video(video_path)
    features = []
    
    for frame in frames:
        results = holistic.process(frame)
        pose = np.array([[lm.x, lm.y, lm.z] for lm in results.pose_landmarks.landmark]) if results.pose_landmarks else np.zeros((33, 3))
        left_hand = np.array([[lm.x, lm.y, lm.z] for lm in results.left_hand_landmarks.landmark]) if results.left_hand_landmarks else np.zeros((21, 3))
        right_hand = np.array([[lm.x, lm.y, lm.z] for lm in results.right_hand_landmarks.landmark]) if results.right_hand_landmarks else np.zeros((21, 3))
        
        frame_features = np.concatenate([pose, left_hand, right_hand]).flatten()
        features.append(frame_features)
    
    holistic.close()
    return np.array(features)

def preprocess_data(csv_path, quick_train=False):
    """
    Loads dataset, extracts features, and preprocesses data.
    If quick_train is True, filters dataset for selected labels.
    """
    df = pd.read_csv(csv_path)

    if quick_train:
        quick_classes = ['loud', 'quiet', 'sick', 'healthy'] #, 'quiet', 'sick', 'healthy'
        df = df[df['Adjectives'].isin(quick_classes)]

    labels = df['Adjectives'].values
    paths = df['Path'].values
    
    features = []
    for path in paths:
        print(f"Processing: {path}")
        features.append(extract_features(path))
    
    # Padding sequences to equal length
    max_length = max(len(f) for f in features)
    features = [np.pad(f, ((0, max_length - len(f)), (0, 0)), mode='constant') for f in features]
    features = np.array(features)
    
    # Encode labels
    label_encoder = LabelEncoder()
    labels_encoded = label_encoder.fit_transform(labels)
    onehot_encoder = OneHotEncoder(sparse_output=False)
    labels_onehot = onehot_encoder.fit_transform(labels_encoded.reshape(-1, 1))
    
    return features, labels_onehot, label_encoder.classes_

def check_dimensions(features, labels, x_train, x_val, y_train, y_val):
    """Prints data dimensions for debugging."""
    print("\n=== Dimension Check ===")
    print(f"Raw features shape: {features.shape}")
    print(f"Raw labels shape: {labels.shape}")
    print(f"Training features shape: {x_train.shape}")
    print(f"Validation features shape: {x_val.shape}")
    print(f"Training labels shape: {y_train.shape}")
    print(f"Validation labels shape: {y_val.shape}")
    print("=======================\n")

class STGCN_LSTM_Model(tf.keras.Model):
    """Spatial-Temporal Graph Convolutional Network + LSTM."""
    def __init__(self, lstm_units, num_classes):
        super(STGCN_LSTM_Model, self).__init__()
        self.conv1 = tf.keras.layers.Conv2D(64, (1, 3), activation='relu')  # Fix kernel shape
        self.conv2 = tf.keras.layers.Conv2D(64, (1, 3), activation='relu')  # Fix kernel shape
        self.flatten = tf.keras.layers.Flatten()
        self.reshape = tf.keras.layers.Reshape((-1, 64))  # Reshape for LSTM
        self.lstm = tf.keras.layers.LSTM(lstm_units, return_sequences=False)
        self.fc = tf.keras.layers.Dense(num_classes, activation='softmax')

    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.conv2(x)
        x = self.flatten(x)
        x = self.reshape(x)
        x = self.lstm(x)
        return self.fc(x)


def plot_metrics(history):
    """Plots accuracy, loss, and MSE graphs."""
    plt.figure(figsize=(12, 4))
    for i, metric in enumerate(['accuracy', 'loss']):
        plt.subplot(1, 3, i + 1)
        plt.plot(history.history[metric], label=f'Train {metric}')
        plt.plot(history.history[f'val_{metric}'], label=f'Val {metric}')
        plt.title(metric.capitalize())
        plt.legend()
    plt.show()

def evaluate_model(model, x_val, y_val, classes):
    """Evaluates the model and plots a confusion matrix."""
    y_pred = model.predict(x_val).argmax(axis=1)
    y_true = y_val.argmax(axis=1)
    
    cm = confusion_matrix(y_true, y_pred)
    ConfusionMatrixDisplay(cm, display_labels=classes).plot()
    plt.show()

# Training Function
def train_and_evaluate(quick_train=False):
    """Trains and evaluates the model."""
    features, labels, classes = preprocess_data(CSV_PATH, quick_train=quick_train)
    
    # Split dataset
    x_train, x_val, y_train, y_val = train_test_split(features, labels, test_size=0.2, random_state=42)
    x_train = x_train[..., np.newaxis]  # Keep this to add the channel dimension
    x_val = x_val[..., np.newaxis]

    
    check_dimensions(features, labels, x_train, x_val, y_train, y_val)
    
    # Initialize and train model
    model = STGCN_LSTM_Model(lstm_units=128, num_classes=labels.shape[1])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    history = model.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=20, batch_size=16)
    model.save(MODEL_SAVE_PATH)
    
    plot_metrics(history)
    evaluate_model(model, x_val, y_val, classes)

# Main
if __name__ == "__main__":
    train_and_evaluate(quick_train=True)
