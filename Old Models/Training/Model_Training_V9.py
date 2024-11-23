import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import cv2
import numpy as np
import pandas as pd
import mediapipe as mp
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (Input, LSTM, Dense, Dropout, BatchNormalization, 
                                   Conv2D, Layer, Reshape, Concatenate, GlobalAveragePooling2D)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib.pyplot as plt

# Feature dimensions setup
pose_feature_dim = 33 * 4  # x, y, z, visibility
hand_feature_dim = 21 * 3  # x, y, z
features_per_frame = pose_feature_dim + (2 * hand_feature_dim)

# Parameters
sequence_length = 100
num_features = 64
batch_size = 16
epochs = 100
learning_rate = 0.001

class TemporalAugmentation:
    """Temporal augmentation techniques for sequence data"""
    
    @staticmethod
    def temporal_crop(sequence, min_length=0.8):
        """Randomly crop a sequence temporally"""
        length = len(sequence)
        crop_length = np.random.randint(int(length * min_length), length + 1)
        start = np.random.randint(0, length - crop_length + 1)
        cropped = sequence[start:start + crop_length]
        
        # Resize back to original length using linear interpolation
        indices = np.linspace(0, len(cropped) - 1, length)
        return np.array([cropped[int(i)] for i in indices])
    
    @staticmethod
    def temporal_mask(sequence, num_masks=2, mask_size=0.1):
        """Apply random temporal masks"""
        masked = sequence.copy()
        length = len(sequence)
        mask_length = int(length * mask_size)
        
        for _ in range(num_masks):
            start = np.random.randint(0, length - mask_length)
            masked[start:start + mask_length] = 0
        
        return masked
    
    @staticmethod
    def temporal_warp(sequence, strength=0.2):
        """Apply random temporal warping"""
        length = len(sequence)
        warp_points = np.random.uniform(-strength, strength, size=length)
        warped_indices = np.linspace(0, length - 1, length) + warp_points
        warped_indices = np.clip(warped_indices, 0, length - 1)
        
        # Interpolate using warped indices
        warped = np.array([
            sequence[int(i)] + (sequence[min(int(i) + 1, length - 1)] - sequence[int(i)]) * (i - int(i))
            for i in warped_indices
        ])
        
        return warped

class STGCNBlock(Layer):
    def __init__(self, out_channels, A_hat, temporal_kernel_size=9):
        super(STGCNBlock, self).__init__()
        self.A_hat = tf.constant(A_hat, dtype=tf.float32)
        self.spatial_conv = Conv2D(out_channels, kernel_size=(1, 1))
        self.temporal_conv = Conv2D(out_channels, kernel_size=(temporal_kernel_size, 1), padding='same')
        self.batch_norm = BatchNormalization()
        
    def call(self, x, training=False):
        # Spatial-temporal convolution
        x = tf.einsum('bsnc,nm->bsmc', x, self.A_hat)
        x = self.spatial_conv(x)
        x = self.temporal_conv(x)
        x = self.batch_norm(x, training=training)
        return tf.nn.relu(x)

def create_adjacency_matrix(num_pose_nodes=33, num_hand_nodes=21):
    """Creates normalized adjacency matrix for pose and hand landmarks"""
    total_nodes = num_pose_nodes + 2 * num_hand_nodes
    A = np.zeros((total_nodes, total_nodes))
    
    # Define pose connections
    pose_connections = [
        (0, 1), (1, 2), (2, 3), (3, 7), (0, 4), (4, 5), (5, 6), (6, 8),
        (9, 10), (11, 12), (11, 13), (13, 15), (15, 17), (15, 19), (15, 21),
        (12, 14), (14, 16), (16, 18), (16, 20), (16, 22), (11, 23), (12, 24),
        (23, 24), (23, 25), (24, 26), (25, 27), (26, 28), (27, 29), (28, 30),
        (29, 31), (30, 32), (27, 31), (28, 32)
    ]
    
    # Define hand connections
    hand_connections = [
        (0, 1), (1, 2), (2, 3), (3, 4),
        (0, 5), (5, 6), (6, 7), (7, 8),
        (0, 9), (9, 10), (10, 11), (11, 12),
        (0, 13), (13, 14), (14, 15), (15, 16),
        (0, 17), (17, 18), (18, 19), (19, 20)
    ]
    
    # Add connections to adjacency matrix
    for i, j in pose_connections:
        A[i, j] = A[j, i] = 1
    
    # Add hand connections for both hands
    for offset in [num_pose_nodes, num_pose_nodes + num_hand_nodes]:
        for i, j in hand_connections:
            A[i + offset, j + offset] = A[j + offset, i + offset] = 1
    
    # Add self-connections
    np.fill_diagonal(A, 1)
    
    # Normalize adjacency matrix
    D = np.sum(A, axis=1)
    D_hat = np.diag(np.power(D, -0.5, where=D != 0))
    A_hat = D_hat @ A @ D_hat
    
    return A_hat

def create_hybrid_model(sequence_length, features_per_frame, num_classes):
    """Creates a hybrid ST-GCN + LSTM model"""
    # Calculate nodes and features
    num_pose_nodes, num_hand_nodes = 33, 21
    total_nodes = num_pose_nodes + 2 * num_hand_nodes
    A_hat = create_adjacency_matrix(num_pose_nodes, num_hand_nodes)
    
    # Input layer
    inputs = Input(shape=(sequence_length, features_per_frame))
    
    # Reshape for ST-GCN
    x_gcn = Reshape((sequence_length, total_nodes, -1))(inputs)
    
    # ST-GCN branch
    x_gcn = STGCNBlock(64, A_hat)(x_gcn)
    x_gcn = STGCNBlock(128, A_hat)(x_gcn)
    x_gcn = STGCNBlock(256, A_hat)(x_gcn)
    x_gcn = GlobalAveragePooling2D()(x_gcn)
    
    # LSTM branch
    x_lstm = LSTM(128, return_sequences=True)(inputs)
    x_lstm = BatchNormalization()(x_lstm)
    x_lstm = LSTM(256)(x_lstm)
    x_lstm = BatchNormalization()(x_lstm)
    
    # Merge branches
    x = Concatenate()([x_gcn, x_lstm])
    
    # Final dense layers
    x = Dense(512, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    x = Dense(256, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    outputs = Dense(num_classes, activation='softmax')(x)
    
    # Create and compile model
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def process_video_with_augmentation(video_path, augment=False):
    """Process video and apply temporal augmentation if specified"""
    # Extract keypoints
    frames_keypoints = []
    cap = cv2.VideoCapture(video_path)
    
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
    
    # Handle sequence length
    if len(frames_keypoints) < sequence_length:
        last_frame = frames_keypoints[-1] if frames_keypoints else np.zeros(features_per_frame)
        frames_keypoints.extend([last_frame] * (sequence_length - len(frames_keypoints)))
    else:
        frames_keypoints = frames_keypoints[:sequence_length]
    
    sequence = np.array(frames_keypoints)
    
    # Apply augmentation if specified
    if augment:
        aug = TemporalAugmentation()
        if np.random.random() < 0.3:
            sequence = aug.temporal_crop(sequence)
        if np.random.random() < 0.3:
            sequence = aug.temporal_mask(sequence)
        if np.random.random() < 0.3:
            sequence = aug.temporal_warp(sequence)
    
    return sequence

def train_model(data_df, class_map, model_save_path):
    """Train the hybrid model with augmentation"""
    # Process videos with augmentation
    X, y = [], []
    for idx, row in data_df.iterrows():
        try:
            print(f"Processing video {idx + 1}/{len(data_df)}")
            # Original sequence
            sequence = process_video_with_augmentation(row['Path'], augment=False)
            X.append(sequence)
            y.append(class_map[row['Adjectives']])
            
            # Augmented sequence
            aug_sequence = process_video_with_augmentation(row['Path'], augment=True)
            X.append(aug_sequence)
            y.append(class_map[row['Adjectives']])
            
        except Exception as e:
            print(f"Error processing video {row['Path']}: {str(e)}")
            continue
    
    X = np.array(X)
    y = to_categorical(y, num_classes=len(class_map))
    
    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Create and train model
    model = create_hybrid_model(sequence_length, features_per_frame, len(class_map))
    
    # Training callbacks
    callbacks = [
        tf.keras.callbacks.EarlyStopping(patience=15, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(factor=0.2, patience=7, min_lr=0.00001),
        tf.keras.callbacks.ModelCheckpoint(model_save_path, save_best_only=True)
    ]
    
    # Train model
    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks
    )
    
    return model, history, X_test, y_test

def plot_results(history, y_test, y_pred, class_labels):
    """Plot training history and confusion matrix"""
    # Training history
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
    
    # Confusion matrix
    plt.figure(figsize=(10, 8))
    conf_matrix = confusion_matrix(np.argmax(y_test, axis=1), np.argmax(y_pred, axis=1))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_labels, yticklabels=class_labels)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.show()

if __name__ == "__main__":
    # Initialize MediaPipe
    mp_holistic = mp.solutions.holistic
    
    # Load and prepare data
    csv_path = "E:\\Deep Learning Project\\data\\INCLUDE_dataset\\Adjectives\\Adjectives_videos.csv"
    model_save_path = "C:\\Users\\visha\\OneDrive\\Desktop\\U21CS010\\VII Semester\\Deep Learning HW\\ISL Test\\adjectives_stgcn_lstm_model_with_interpolation.keras"

    # Load data
    data_df = pd.read_csv(csv_path)
    class_labels = sorted(data_df['Adjectives'].unique())
    class_map = {label: idx for idx, label in enumerate(class_labels)}
    
    # Train model
    model, history, X_test, y_test = train_model(data_df, class_map, model_save_path)
    
    # Generate predictions
    y_pred = model.predict(X_test)
    
    # Plot results
    plot_results(history, y_test, y_pred, class_labels)