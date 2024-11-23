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
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

# Parameters
sequence_length = 100
pose_feature_dim = 33 * 4  # x, y, z, visibility
hand_feature_dim = 21 * 3  # x, y, z
features_per_frame = pose_feature_dim + (2 * hand_feature_dim)
batch_size = 32  # Increased batch size for better stability
epochs = 50  # Increased epochs since we'll use early stopping

def speed_augment(sequence, speed_factor):
    """
    Augment sequence by changing its speed
    speed_factor < 1: slower
    speed_factor > 1: faster
    """
    # Calculate new sequence length
    new_length = int(len(sequence) * (1/speed_factor))
    
    # Create new time points
    old_indices = np.arange(len(sequence))
    new_indices = np.linspace(0, len(sequence)-1, new_length)
    
    # Interpolate for each feature
    augmented = np.zeros((new_length, sequence.shape[1]))
    for i in range(sequence.shape[1]):
        augmented[:, i] = np.interp(new_indices, old_indices, sequence[:, i])
    
    # Ensure we return exactly sequence_length frames
    if len(augmented) >= sequence_length:
        return augmented[:sequence_length]
    else:
        # Pad with last frame if needed
        padding = np.tile(augmented[-1:], (sequence_length - len(augmented), 1))
        return np.vstack([augmented, padding])

def extract_keypoints(results):
    """Extract keypoints from MediaPipe results"""
    pose = np.zeros(pose_feature_dim)
    left_hand = np.zeros(hand_feature_dim)
    right_hand = np.zeros(hand_feature_dim)
    
    if results.pose_landmarks:
        pose = np.array([[lm.x, lm.y, lm.z, lm.visibility] 
                        for lm in results.pose_landmarks.landmark]).flatten()
    
    if results.left_hand_landmarks:
        left_hand = np.array([[lm.x, lm.y, lm.z] 
                            for lm in results.left_hand_landmarks.landmark]).flatten()
    
    if results.right_hand_landmarks:
        right_hand = np.array([[lm.x, lm.y, lm.z] 
                             for lm in results.right_hand_landmarks.landmark]).flatten()
    
    return np.concatenate([pose, left_hand, right_hand])

def process_video(video_path, augment=False):
    """Process video and optionally apply speed augmentation"""
    frames_keypoints = []
    cap = cv2.VideoCapture(video_path)
    
    with mp.solutions.holistic.Holistic(
        min_detection_confidence=0.7, 
        min_tracking_confidence=0.7
    ) as holistic:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = holistic.process(frame_rgb)
            keypoints = extract_keypoints(results)
            frames_keypoints.append(keypoints)
    
    cap.release()
    
    if len(frames_keypoints) == 0:
        return np.zeros((sequence_length, features_per_frame))
    
    sequence = np.array(frames_keypoints)
    
    if augment:
        # Randomly choose speed factor between 0.5 (half speed) and 1.5 (1.5x speed)
        speed_factor = np.random.uniform(0.5, 1.5)
        sequence = speed_augment(sequence, speed_factor)
    
    # Ensure final sequence length
    if len(sequence) >= sequence_length:
        return sequence[:sequence_length]
    else:
        padding = np.tile(sequence[-1:], (sequence_length - len(sequence), 1))
        return np.vstack([sequence, padding])

def create_model(input_shape, num_classes):
    """Create LSTM model with additional regularization"""
    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=input_shape,
             kernel_regularizer=tf.keras.regularizers.l2(0.01)),  # Reduced complexity, added L2
        BatchNormalization(),
        Dropout(0.4),  # Increased dropout
        LSTM(32, kernel_regularizer=tf.keras.regularizers.l2(0.01)),  # Reduced complexity
        BatchNormalization(),
        Dropout(0.4),  # Increased dropout
        Dense(16, activation='relu',  # Reduced complexity
              kernel_regularizer=tf.keras.regularizers.l2(0.01)),
        BatchNormalization(),
        Dropout(0.4),
        Dense(num_classes, activation='softmax')
    ])
    
    model.compile(
        optimizer=Adam(learning_rate=0.0005),  # Reduced learning rate
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model


def prepare_data(data_df, target_words=['loud', 'quiet', 'sick', 'healthy']):
    """Prepare dataset with specified target words"""
    # Filter for target words
    filtered_df = data_df[data_df['Adjectives'].isin(target_words)]
    
    X, y = [], []
    for _, row in filtered_df.iterrows():
        try:
            # Process original sequence
            sequence = process_video(row['Path'])
            X.append(sequence)
            y.append(target_words.index(row['Adjectives']))
            
            # Add augmented sequence
            aug_sequence = process_video(row['Path'], augment=True)
            X.append(aug_sequence)
            y.append(target_words.index(row['Adjectives']))
            
        except Exception as e:
            print(f"Error processing {row['Path']}: {str(e)}")
            continue
    
    X = np.array(X)
    y = to_categorical(y, num_classes=len(target_words))
    
    return train_test_split(X, y, test_size=0.2, random_state=42)

def plot_confusion_matrix(y_true, y_pred, target_words):
    """Plot confusion matrix"""
    # Convert one-hot encoded labels back to class indices
    y_true_labels = np.argmax(y_true, axis=1)
    y_pred_labels = np.argmax(y_pred, axis=1)
    
    # Create confusion matrix
    cm = confusion_matrix(y_true_labels, y_pred_labels)
    
    # Plot
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=target_words,
                yticklabels=target_words)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.show()
    
    # Print classification report
    print("\nClassification Report:")
    print(classification_report(y_true_labels, y_pred_labels, target_names=target_words))

def train_and_evaluate(X_train, X_test, y_train, y_test, target_words):
    """Train and evaluate the model"""
    model = create_model((sequence_length, features_per_frame), len(target_words))
    
    # Print model summary
    print("\nModel Summary:")
    model.summary()
    
    # Add class weights to handle imbalanced data
    class_weights = dict(enumerate(
        len(y_train) / (len(target_words) * np.sum(y_train, axis=0))
    ))
    
    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=epochs,
        batch_size=batch_size,
        class_weight=class_weights,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=15,
                restore_best_weights=True
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=0.00001
            )
        ]
    )
    
    # Plot training history
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training')
    plt.plot(history.history['val_accuracy'], label='Validation')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training')
    plt.plot(history.history['val_loss'], label='Validation')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()
    
        # Generate and plot confusion matrix
    y_pred = model.predict(X_test)
    plot_confusion_matrix(y_test, y_pred, target_words)
    
    return model, history

if __name__ == "__main__":
    # Load data
    csv_path = "E:\\Deep Learning Project\\data\\INCLUDE_dataset\\Adjectives\\Adjectives_videos.csv"
    model_save_path = "C:\\Users\\visha\\OneDrive\\Desktop\\U21CS010\\VII Semester\\Deep Learning HW\\ISL Test\\adjectives_stgcn_lstm_model_with_interpolation.keras"
    data_df = pd.read_csv(csv_path)
    
    # Target words
    target_words = ['loud', 'quiet', 'sick', 'healthy']
    
    # Prepare data
    X_train, X_test, y_train, y_test = prepare_data(data_df, target_words)
    
    # Train model
    model, history = train_and_evaluate(X_train, X_test, y_train, y_test, target_words)