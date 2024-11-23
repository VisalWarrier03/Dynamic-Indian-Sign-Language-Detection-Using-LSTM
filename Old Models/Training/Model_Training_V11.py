import mediapipe as mp
import numpy as np
import pandas as pd
import cv2
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# Constants
CSV_PATH = r"Adjectives_videos.csv"
MODEL_SAVE_PATH = r"adjectives_stgcn_lstm_model.keras"

# Helper functions
def read_video(video_path):
    """Reads a video and returns its frames as a list of images."""
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

def extract_mediapipe_features(video_path):
    """
    Extracts skeletal features (pose, left hand, right hand) using MediaPipe.
    """
    import mediapipe as mp
    import numpy as np

    mp_holistic = mp.solutions.holistic
    holistic = mp_holistic.Holistic(static_image_mode=False)
    
    features = []
    frames = read_video(video_path)  # Utility function to extract video frames
    
    for frame in frames:
        results = holistic.process(frame)
        
        # Pose landmarks (33 keypoints)
        if results.pose_landmarks:
            pose_features = [[lm.x, lm.y, lm.z, lm.visibility] for lm in results.pose_landmarks.landmark]
        else:
            pose_features = [[0, 0, 0, 0]] * 33  # Default for missing pose landmarks

        # Left hand landmarks (21 keypoints)
        if results.left_hand_landmarks:
            left_hand_features = [[lm.x, lm.y, lm.z] for lm in results.left_hand_landmarks.landmark]
        else:
            left_hand_features = [[0, 0, 0]] * 21  # Default for missing left hand landmarks

        # Right hand landmarks (21 keypoints)
        if results.right_hand_landmarks:
            right_hand_features = [[lm.x, lm.y, lm.z] for lm in results.right_hand_landmarks.landmark]
        else:
            right_hand_features = [[0, 0, 0]] * 21  # Default for missing right hand landmarks

        # Combine features: Pose + Left Hand + Right Hand
        frame_features = np.concatenate([
            np.array(pose_features).flatten(),  # Shape: (33 * 4,)
            np.array(left_hand_features).flatten(),  # Shape: (21 * 3,)
            np.array(right_hand_features).flatten()  # Shape: (21 * 3,)
        ])
        features.append(frame_features)
    
    holistic.close()
    return np.array(features)  # Shape: (num_frames, 258)


def preprocess_data(csv_path, quick_train=False):
    """
    Loads dataset from CSV and extracts features. Filters for quick training if enabled.
    """
    df = pd.read_csv(csv_path)

    # Filter for quick training
    if quick_train:
        quick_words = ['loud', 'quiet', 'sick', 'healthy'] #
        df = df[df['Adjectives'].isin(quick_words)]

    labels = df['Adjectives'].values
    paths = df['Path'].values
    
    features = []
    for path in paths:
        print(f"Processing: {path}")
        features.append(extract_mediapipe_features(path))
    
    # Padding to make all sequences of equal length
    max_length = max(len(f) for f in features)
    features = [np.pad(f, ((0, max_length - len(f)), (0, 0)), mode='constant') for f in features]
    features = np.array(features)
    
    # Label encoding
    label_encoder = LabelEncoder()
    labels_encoded = label_encoder.fit_transform(labels)
    onehot_encoder = OneHotEncoder(sparse_output=False)
    labels_onehot = onehot_encoder.fit_transform(labels_encoded.reshape(-1, 1))
    
    return features, labels_onehot, label_encoder.classes_

class STGCN_LSTM(tf.keras.Model):
    def __init__(self, lstm_units, num_classes):
        super(STGCN_LSTM, self).__init__()
        self.conv1 = tf.keras.layers.Conv2D(64, kernel_size=(1, 1), activation='relu')
        self.conv2 = tf.keras.layers.Conv2D(64, kernel_size=(1, 1), activation='relu')
        self.flatten = tf.keras.layers.Flatten()
        self.reshape = tf.keras.layers.Reshape((-1, 64))  # Reshape for LSTM
        self.lstm = tf.keras.layers.LSTM(lstm_units, return_sequences=False)
        self.fc = tf.keras.layers.Dense(num_classes, activation='softmax')

    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.conv2(x)
        x = self.flatten(x)
        x = self.reshape(x)  # Ensure compatibility with LSTM
        x = self.lstm(x)
        return self.fc(x)


    
def check_dimensions(features, labels, x_train, x_val, y_train, y_val):
    """
    Prints the dimensions of the data at various stages to identify issues.
    """
    print("===== Dimension Check =====")
    print(f"Raw features shape: {features.shape}")  # Shape of preprocessed features
    print(f"Raw labels shape: {labels.shape}")      # Shape of labels
    
    print(f"Training features shape: {x_train.shape}")
    print(f"Validation features shape: {x_val.shape}")
    
    print(f"Training labels shape: {y_train.shape}")
    print(f"Validation labels shape: {y_val.shape}")
    
    # Check the shape passed to the model
    input_shape = x_train[0].shape
    print(f"Model input shape (single sample): {input_shape}")
    print("===========================")


# Training and evaluation
def train_and_evaluate(quick_train=False):
    """
    Trains and evaluates the model. If quick_train is True, only use a subset of the dataset.
    """
    features, labels, classes = preprocess_data(CSV_PATH, quick_train=quick_train)
    
    # Split dataset
    x_train, x_val, y_train, y_val = train_test_split(features, labels, test_size=0.2, random_state=42)
    
    # Reshape inputs for Conv2D
    x_train = x_train[..., np.newaxis]
    x_val = x_val[..., np.newaxis]
    
    # Validate dimensions
    check_dimensions(features, labels, x_train, x_val, y_train, y_val)
    
    # Initialize and compile model
    model = STGCN_LSTM(lstm_units=128, num_classes=len(classes))

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy', 'mse'])
    
    # Train
    history = model.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=20, batch_size=16)
    
    # Save model
    model.save(MODEL_SAVE_PATH)
    
    # Plot metrics
    plot_metrics(history)
    
    # Evaluate
    evaluate_model(model, x_val, y_val, classes)



def plot_metrics(history):
    """Plots accuracy, loss, and MSE graphs."""
    plt.figure(figsize=(12, 4))
    for i, metric in enumerate(['accuracy', 'loss', 'mse']):
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

# Run training and evaluation
if __name__ == "__main__":
    train_and_evaluate(True)
