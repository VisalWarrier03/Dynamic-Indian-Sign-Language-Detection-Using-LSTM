import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import cv2
import numpy as np
import pandas as pd
import mediapipe as mp
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Conv1D, MaxPooling1D, BatchNormalization
from tensorflow.keras.layers import Input, Concatenate, GlobalAveragePooling1D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import pickle
from pathlib import Path

# Initialize MediaPipe Holistic
mp_holistic = mp.solutions.holistic

# File paths
csv_path = "E:\\Deep Learning Project\\data\\INCLUDE_dataset\\Adjectives\\Adjectives_videos.csv"
model_save_path = "C:\\Users\\visha\\OneDrive\\Desktop\\U21CS010\\VII Semester\\Deep Learning HW\\ISL Test\\adjectives_cnn_lstm_model.keras"

# Verify CSV file exists
if not os.path.exists(csv_path):
    raise FileNotFoundError(f"CSV file not found at path: {csv_path}")

# Load CSV and extract classes with error handling
try:
    data_df = pd.read_csv(csv_path)
    if data_df.empty:
        raise ValueError("CSV file is empty")
    if 'Adjectives' not in data_df.columns or 'Path' not in data_df.columns:
        raise ValueError("CSV file must contain 'Adjectives' and 'Path' columns")
        
    class_labels = data_df['Adjectives'].unique().tolist()
    num_classes = len(class_labels)
    if num_classes == 0:
        raise ValueError("No unique classes found in the dataset")
    
    class_map = {label: index for index, label in enumerate(class_labels)}
    print(f"Found {num_classes} unique classes: {class_labels}")
except Exception as e:
    raise Exception(f"Error loading CSV file: {str(e)}")

# Parameters
sequence_length = 200
hand_feature_dim = 21 * 3
pose_feature_dim = 33 * 4
features_per_frame = pose_feature_dim + (hand_feature_dim * 2)
batch_size = 16
epochs = 50

def extract_keypoints(results):
    """Extract keypoints with error handling and validation"""
    try:
        # Initialize arrays
        pose = np.zeros(pose_feature_dim, dtype=np.float32)
        left_hand = np.zeros(hand_feature_dim, dtype=np.float32)
        right_hand = np.zeros(hand_feature_dim, dtype=np.float32)
        
        # Extract pose landmarks with relative position to nose
        if results.pose_landmarks:
            nose = np.array([
                results.pose_landmarks.landmark[0].x,
                results.pose_landmarks.landmark[0].y,
                results.pose_landmarks.landmark[0].z
            ])
            
            for idx, landmark in enumerate(results.pose_landmarks.landmark):
                pose[idx * 4: (idx + 1) * 4] = [
                    landmark.x - nose[0],
                    landmark.y - nose[1],
                    landmark.z - nose[2],
                    landmark.visibility
                ]
        
        # Extract hand landmarks relative to wrist
        if results.left_hand_landmarks:
            wrist = np.array([
                results.left_hand_landmarks.landmark[0].x,
                results.left_hand_landmarks.landmark[0].y,
                results.left_hand_landmarks.landmark[0].z
            ])
            
            for idx, landmark in enumerate(results.left_hand_landmarks.landmark):
                left_hand[idx * 3: (idx + 1) * 3] = [
                    landmark.x - wrist[0],
                    landmark.y - wrist[1],
                    landmark.z - wrist[2]
                ]
        
        if results.right_hand_landmarks:
            wrist = np.array([
                results.right_hand_landmarks.landmark[0].x,
                results.right_hand_landmarks.landmark[0].y,
                results.right_hand_landmarks.landmark[0].z
            ])
            
            for idx, landmark in enumerate(results.right_hand_landmarks.landmark):
                right_hand[idx * 3: (idx + 1) * 3] = [
                    landmark.x - wrist[0],
                    landmark.y - wrist[1],
                    landmark.z - wrist[2]
                ]
        
        # Validate output shape
        output = np.concatenate([pose, left_hand, right_hand])
        if output.shape[0] != features_per_frame:
            raise ValueError(f"Invalid feature shape. Expected {features_per_frame}, got {output.shape[0]}")
        
        return output
    
    except Exception as e:
        print(f"Error in extract_keypoints: {str(e)}")
        return np.zeros(features_per_frame, dtype=np.float32)

def process_video(video_path):
    """Process video with enhanced error handling and validation"""
    try:
        if not isinstance(video_path, str):
            raise ValueError(f"Invalid video path type: {type(video_path)}")
        
        if not os.path.isfile(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")
        
        frames_keypoints = []
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise ValueError(f"Failed to open video: {video_path}")
        
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if frame_count == 0:
            raise ValueError(f"Video contains no frames: {video_path}")
        
        print(f"Processing video: {video_path}")
        print(f"Total frames: {frame_count}")
        
        with mp_holistic.Holistic(
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7,
            model_complexity=2
        ) as holistic:
            frame_idx = 0
            while cap.isOpened() and len(frames_keypoints) < sequence_length:
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_idx += 1
                if frame_idx % 10 == 0:  # Progress update every 10 frames
                    print(f"Processing frame {frame_idx}/{frame_count}")
                
                frame = cv2.resize(frame, (960, 540))
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = holistic.process(frame_rgb)
                
                keypoints = extract_keypoints(results)
                if not np.all(keypoints == 0):  # Only add frame if keypoints were detected
                    frames_keypoints.append(keypoints)
        
        cap.release()
        
        frames_keypoints = np.array(frames_keypoints, dtype=np.float32)
        
        if len(frames_keypoints) == 0:
            raise ValueError(f"No valid keypoints detected in video: {video_path}")
        
        print(f"Extracted {len(frames_keypoints)} frames with keypoints")
        
        # Handle sequence length
        if len(frames_keypoints) < sequence_length:
            print(f"Padding sequence from {len(frames_keypoints)} to {sequence_length} frames")
            indices = np.linspace(0, len(frames_keypoints)-1, sequence_length)
            frames_keypoints = np.array([frames_keypoints[int(i)] for i in indices])
        elif len(frames_keypoints) > sequence_length:
            print(f"Downsampling sequence from {len(frames_keypoints)} to {sequence_length} frames")
            indices = np.linspace(0, len(frames_keypoints)-1, sequence_length, dtype=int)
            frames_keypoints = frames_keypoints[indices]
        
        # Validate final output shape
        if frames_keypoints.shape != (sequence_length, features_per_frame):
            raise ValueError(f"Invalid output shape. Expected {(sequence_length, features_per_frame)}, got {frames_keypoints.shape}")
        
        return frames_keypoints
    
    except Exception as e:
        print(f"Error processing video {video_path}: {str(e)}")
        return None

# Keep your existing model architecture

# Prepare dataset with enhanced error handling
print("\nProcessing videos and extracting features...")
X, y = [], []
successful_videos = 0
failed_videos = 0

for index, row in data_df.iterrows():
    print(f"\nProcessing video {index + 1}/{len(data_df)}")
    video_path = row['Path']
    label = class_map[row['Adjectives']]
    
    try:
        processed_sequence = process_video(video_path)
        if processed_sequence is not None:
            X.append(processed_sequence)
            y.append(label)
            successful_videos += 1
            print(f"Successfully processed video {video_path}")
        else:
            failed_videos += 1
            print(f"Failed to process video {video_path}")
    except Exception as e:
        failed_videos += 1
        print(f"Error processing video {video_path}: {str(e)}")
        continue

print(f"\nProcessing complete:")
print(f"Successfully processed videos: {successful_videos}")
print(f"Failed videos: {failed_videos}")

if len(X) == 0:
    raise ValueError("No videos were successfully processed. Please check the video paths and data.")

X = np.array(X, dtype=np.float32)
y = to_categorical(y, num_classes=num_classes)

print(f"\nFinal dataset shape:")
print(f"X shape: {X.shape}")
print(f"y shape: {y.shape}")

# Split dataset with validation
if len(X) < 2:  # Need at least 2 samples to split
    raise ValueError(f"Not enough samples to split. Got {len(X)} samples, need at least 2.")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"\nTrain/Test split:")
print(f"X_train shape: {X_train.shape}")
print(f"X_test shape: {X_test.shape}")

# Continue with your existing training code...

def create_lstm_cnn_model(sequence_length, features_per_frame, num_classes):
    """
    Create a hybrid LSTM-CNN model with separate branches for pose and hand features
    """
    # Input shapes
    pose_input = Input(shape=(sequence_length, pose_feature_dim))
    hand_input = Input(shape=(sequence_length, hand_feature_dim * 2))
    
    # LSTM branch for pose features
    pose_lstm = LSTM(128, return_sequences=True)(pose_input)
    pose_lstm = BatchNormalization()(pose_lstm)
    pose_lstm = Dropout(0.3)(pose_lstm)
    
    # CNN branch for pose features
    pose_conv = Conv1D(64, kernel_size=5, padding='same', activation='relu')(pose_lstm)
    pose_conv = BatchNormalization()(pose_conv)
    pose_conv = MaxPooling1D(pool_size=2)(pose_conv)
    pose_conv = Conv1D(128, kernel_size=5, padding='same', activation='relu')(pose_conv)
    pose_conv = BatchNormalization()(pose_conv)
    pose_conv = MaxPooling1D(pool_size=2)(pose_conv)
    
    # LSTM branch for hand features
    hand_lstm = LSTM(128, return_sequences=True)(hand_input)
    hand_lstm = BatchNormalization()(hand_lstm)
    hand_lstm = Dropout(0.3)(hand_lstm)
    
    # CNN branch for hand features
    hand_conv = Conv1D(64, kernel_size=5, padding='same', activation='relu')(hand_lstm)
    hand_conv = BatchNormalization()(hand_conv)
    hand_conv = MaxPooling1D(pool_size=2)(hand_conv)
    hand_conv = Conv1D(128, kernel_size=5, padding='same', activation='relu')(hand_conv)
    hand_conv = BatchNormalization()(hand_conv)
    hand_conv = MaxPooling1D(pool_size=2)(hand_conv)
    
    # Merge pose and hand branches
    merged = Concatenate()([pose_conv, hand_conv])
    
    # Final LSTM layers
    x = LSTM(256, return_sequences=True)(merged)
    x = BatchNormalization()(x)
    x = Dropout(0.4)(x)
    x = LSTM(128)(x)
    x = BatchNormalization()(x)
    
    # Dense layers for classification
    x = Dense(256, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.4)(x)
    x = Dense(128, activation='relu')(x)
    x = BatchNormalization()(x)
    outputs = Dense(num_classes, activation='softmax')(x)
    
    # Create model
    model = Model(inputs=[pose_input, hand_input], outputs=outputs)
    return model

def prepare_split_inputs(X):
    """Split the input features into pose and hand components"""
    pose_features = X[:, :, :pose_feature_dim]
    hand_features = X[:, :, pose_feature_dim:]
    return [pose_features, hand_features]

# Training setup
def train_model(X_train, X_test, y_train, y_test, sequence_length, batch_size=16, epochs=50):
    """Train the model with proper callbacks and monitoring"""
    # Prepare split inputs
    X_train_split = prepare_split_inputs(X_train)
    X_test_split = prepare_split_inputs(X_test)
    
    # Create model
    model = create_lstm_cnn_model(sequence_length, features_per_frame, num_classes)
    
    # Compile model
    optimizer = Adam(learning_rate=0.0001, clipnorm=1.0)
    model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Print model summary
    model.summary()
    
    # Calculate class weights for imbalanced dataset
    class_weights = dict(enumerate(np.bincount(np.argmax(y_train, axis=1))))
    total = sum(class_weights.values())
    class_weights = {k: total/(v * len(class_weights)) for k, v in class_weights.items()}
    
    # Callbacks
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=15,
        restore_best_weights=True,
        verbose=1
    )
    
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.2,
        patience=7,
        min_lr=0.00001,
        verbose=1
    )
    
    # Model checkpoint
    checkpoint_path = "best_model_checkpoint.keras"
    model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
        checkpoint_path,
        monitor='val_accuracy',
        save_best_only=True,
        mode='max',
        verbose=1
    )
    
    # Train model
    print("\nStarting model training...")
    history = model.fit(
        X_train_split,
        y_train,
        validation_data=(X_test_split, y_test),
        epochs=epochs,
        batch_size=batch_size,
        class_weight=class_weights,
        callbacks=[early_stopping, reduce_lr, model_checkpoint],
        verbose=1
    )
    
    # Load best model
    best_model = tf.keras.models.load_model(checkpoint_path)
    
    # Evaluate model
    print("\nEvaluating model on test set...")
    test_loss, test_accuracy = best_model.evaluate(X_test_split, y_test, verbose=0)
    print(f"Test accuracy: {test_accuracy:.4f}")
    print(f"Test loss: {test_loss:.4f}")
    
    return best_model, history

# Main training execution
print("\nPreparing to train model...")
print(f"Input shape: {X_train.shape}")
print(f"Number of classes: {num_classes}")
print(f"Sequence length: {sequence_length}")

# Train the model
model, history = train_model(
    X_train,
    X_test, 
    y_train,
    y_test,
    sequence_length,
    batch_size=batch_size,
    epochs=epochs
)

# Save the final model
model.save('Adj_sign_language_model.keras')

print(model.summary())

# Plot training history
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 4))

# Plot training & validation accuracy
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

# Plot training & validation loss
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.show()