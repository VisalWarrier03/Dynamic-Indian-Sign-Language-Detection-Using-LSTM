import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
import json
from datetime import datetime
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import mediapipe as mp
import cv2
from sklearn.preprocessing import LabelEncoder, OneHotEncoder





# Constants
CSV_PATH = "Adjectives_videos.csv"
BASE_DIR = "ISL Test\Final_Run"
MAX_SEQUENCE_LENGTH = 150
FEATURE_DIM = 225

# Create base directory
os.makedirs(BASE_DIR, exist_ok=True)

# Initialize MediaPipe
mp_holistic = mp.solutions.holistic

# Helper Functions from first part (keeping them as is)
def read_video(video_path):
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

def interpolate_missing_frames(landmarks):
    for i in range(landmarks.shape[1]):
        column = landmarks[:, i]
        nan_indices = np.isnan(column)
        if np.any(nan_indices):
            not_nan = np.where(~nan_indices)[0]
            if not_nan.size > 0:
                column[nan_indices] = np.interp(np.where(nan_indices)[0], not_nan, column[not_nan])
            else:
                column[:] = 0
        landmarks[:, i] = column
    return landmarks

def extract_features(video_path):
    holistic = mp_holistic.Holistic(static_image_mode=False)
    frames = read_video(video_path)
    video_data = []

    for frame in frames:
        results = holistic.process(frame)

        pose = np.array([[lm.x, lm.y, lm.z] for lm in results.pose_landmarks.landmark]) if results.pose_landmarks else np.zeros((33, 3))
        left_hand = np.array([[lm.x, lm.y, lm.z] for lm in results.left_hand_landmarks.landmark]) if results.left_hand_landmarks else np.zeros((21, 3))
        right_hand = np.array([[lm.x, lm.y, lm.z] for lm in results.right_hand_landmarks.landmark]) if results.right_hand_landmarks else np.zeros((21, 3))

        if pose.shape != (33, 3):
            pose = np.zeros((33, 3))
        if left_hand.shape != (21, 3):
            left_hand = np.zeros((21, 3))
        if right_hand.shape != (21, 3):
            right_hand = np.zeros((21, 3))

        frame_features = np.concatenate([pose.flatten(), left_hand.flatten(), right_hand.flatten()])
        video_data.append(frame_features)

    holistic.close()
    
    video_data = np.array(video_data)
    video_data = interpolate_missing_frames(video_data)
    return video_data

def preprocess_data(csv_path, quick_train=False):
    df = pd.read_csv(csv_path)

    if quick_train:
        quick_classes = ['loud', 'quiet', 'sick', 'healthy']
        df = df[df['Adjectives'].isin(quick_classes)]

    class_counts = df['Adjectives'].value_counts()
    valid_classes = class_counts[class_counts >= 18].index
    df = df[df['Adjectives'].isin(valid_classes)]

    print(f"Classes retained after filtering: {valid_classes.tolist()}")
    print(f"Total samples after filtering: {len(df)}")

    labels = df['Adjectives'].values
    paths = df['Path'].values

    features = []
    for path in paths:
        print(f"Processing: {path}")
        video_features = extract_features(path)

        if video_features.shape[0] < MAX_SEQUENCE_LENGTH:
            padding = np.zeros((MAX_SEQUENCE_LENGTH - video_features.shape[0], FEATURE_DIM))
            video_features = np.vstack([video_features, padding])
        elif video_features.shape[0] > MAX_SEQUENCE_LENGTH:
            video_features = video_features[:MAX_SEQUENCE_LENGTH]

        features.append(video_features)

    features = np.array(features)

    label_encoder = LabelEncoder()
    labels_encoded = label_encoder.fit_transform(labels)
    onehot_encoder = OneHotEncoder(sparse_output=False)
    labels_onehot = onehot_encoder.fit_transform(labels_encoded.reshape(-1, 1))

    return features, labels_onehot, label_encoder.classes_

def create_model_variants(input_shape, num_classes):
    """Creates different model variants optimized for keypoint processing."""
    models = {}
    
    # Common initial layers for all models
    def get_input_layers():
        input_layer = tf.keras.layers.Input(shape=input_shape)
        mask_layer = tf.keras.layers.Masking(mask_value=0.0)(input_layer)
        return input_layer, mask_layer
    
    # Original model
    input_layer, mask_layer = get_input_layers()
    lstm1 = tf.keras.layers.LSTM(64, return_sequences=True, activation='tanh')(mask_layer)
    lstm2 = tf.keras.layers.LSTM(64, return_sequences=False, activation='tanh')(lstm1)
    dense1 = tf.keras.layers.Dense(64, activation='tanh')(lstm2)
    dropout = tf.keras.layers.Dropout(0.2)(dense1)
    output = tf.keras.layers.Dense(num_classes, activation='softmax')(dropout)
    models["original"] = tf.keras.Model(inputs=input_layer, outputs=output)
    
    # Increased hidden states 1
    input_layer, mask_layer = get_input_layers()
    lstm1 = tf.keras.layers.LSTM(128, return_sequences=True, activation='tanh')(mask_layer)
    lstm2 = tf.keras.layers.LSTM(128, return_sequences=False, activation='tanh')(lstm1)
    dense1 = tf.keras.layers.Dense(64, activation='tanh')(lstm2)
    dropout = tf.keras.layers.Dropout(0.2)(dense1)
    output = tf.keras.layers.Dense(num_classes, activation='softmax')(dropout)
    models["increased_hidden_1"] = tf.keras.Model(inputs=input_layer, outputs=output)
    
    # Increased hidden states 2
    input_layer, mask_layer = get_input_layers()
    lstm1 = tf.keras.layers.LSTM(256, return_sequences=True, activation='tanh')(mask_layer)
    lstm2 = tf.keras.layers.LSTM(256, return_sequences=False, activation='tanh')(lstm1)
    dense1 = tf.keras.layers.Dense(64, activation='tanh')(lstm2)
    dropout = tf.keras.layers.Dropout(0.2)(dense1)
    output = tf.keras.layers.Dense(num_classes, activation='softmax')(dropout)
    models["increased_hidden_2"] = tf.keras.Model(inputs=input_layer, outputs=output)
    
    # Decreased hidden states 1
    input_layer, mask_layer = get_input_layers()
    lstm1 = tf.keras.layers.LSTM(32, return_sequences=True, activation='tanh')(mask_layer)
    lstm2 = tf.keras.layers.LSTM(32, return_sequences=False, activation='tanh')(lstm1)
    dense1 = tf.keras.layers.Dense(64, activation='tanh')(lstm2)
    dropout = tf.keras.layers.Dropout(0.2)(dense1)
    output = tf.keras.layers.Dense(num_classes, activation='softmax')(dropout)
    models["decreased_hidden_1"] = tf.keras.Model(inputs=input_layer, outputs=output)
    
    # Decreased hidden states 2
    input_layer, mask_layer = get_input_layers()
    lstm1 = tf.keras.layers.LSTM(16, return_sequences=True, activation='tanh')(mask_layer)
    lstm2 = tf.keras.layers.LSTM(16, return_sequences=False, activation='tanh')(lstm1)
    dense1 = tf.keras.layers.Dense(64, activation='tanh')(lstm2)
    dropout = tf.keras.layers.Dropout(0.2)(dense1)
    output = tf.keras.layers.Dense(num_classes, activation='softmax')(dropout)
    models["decreased_hidden_2"] = tf.keras.Model(inputs=input_layer, outputs=output)
    
    # Different activations and dropouts
    activation_dropout_configs = [
        ('relu', 0.3),
        ('relu', 0.5),
        ('elu', 0.2),
        ('elu', 0.4),
        ('leaky_relu', 0.2),
        ('leaky_relu', 0.4)
    ]
    
    for i, (act, drop) in enumerate(activation_dropout_configs):
        input_layer, mask_layer = get_input_layers()
        lstm1 = tf.keras.layers.LSTM(64, return_sequences=True, activation=act)(mask_layer)
        lstm2 = tf.keras.layers.LSTM(64, return_sequences=False, activation=act)(lstm1)
        dense1 = tf.keras.layers.Dense(64, activation=act)(lstm2)
        dropout = tf.keras.layers.Dropout(drop)(dense1)
        output = tf.keras.layers.Dense(num_classes, activation='softmax')(dropout)
        models[f"act_drop_{i+1}"] = tf.keras.Model(inputs=input_layer, outputs=output)
    
    # Cascaded LSTM
    input_layer, mask_layer = get_input_layers()
    lstm1 = tf.keras.layers.LSTM(96, return_sequences=True)(mask_layer)
    lstm2 = tf.keras.layers.LSTM(64, return_sequences=True)(lstm1)
    lstm3 = tf.keras.layers.LSTM(32, return_sequences=False)(lstm2)
    dense1 = tf.keras.layers.Dense(64, activation='tanh')(lstm3)
    dropout = tf.keras.layers.Dropout(0.2)(dense1)
    output = tf.keras.layers.Dense(num_classes, activation='softmax')(dropout)
    models["cascaded_lstm"] = tf.keras.Model(inputs=input_layer, outputs=output)
    
    # Custom Attention Layer with masking support
    class MaskedAttentionLayer(tf.keras.layers.Layer):
        def __init__(self, **kwargs):
            super(MaskedAttentionLayer, self).__init__(**kwargs)
            
        def build(self, input_shape):
            self.W = self.add_weight(name="att_weight",
                                   shape=(input_shape[-1], 1),
                                   initializer="normal")
            super(MaskedAttentionLayer, self).build(input_shape)
            
        def call(self, x, mask=None):
            e = tf.keras.backend.tanh(tf.keras.backend.dot(x, self.W))
            
            if mask is not None:
                mask = tf.cast(mask, dtype=tf.float32)
                mask = tf.expand_dims(mask, axis=-1)
                e = e * mask - 1e10 * (1 - mask)
                
            a = tf.keras.backend.softmax(e, axis=1)
            output = x * a
            return tf.keras.backend.sum(output, axis=1)
        
        def compute_mask(self, inputs, mask=None):
            return None
    
    # Attention LSTM
    input_layer, mask_layer = get_input_layers()
    lstm_out = tf.keras.layers.LSTM(64, return_sequences=True)(mask_layer)
    attention_out = MaskedAttentionLayer()(lstm_out)
    dense1 = tf.keras.layers.Dense(64, activation='tanh')(attention_out)
    dropout = tf.keras.layers.Dropout(0.2)(dense1)
    output = tf.keras.layers.Dense(num_classes, activation='softmax')(dropout)
    models["attention_lstm"] = tf.keras.Model(inputs=input_layer, outputs=output)
    
    # Custom Pooling Layer with masking support
    class MaskedPooling(tf.keras.layers.Layer):
        def __init__(self, pooling_type='max', **kwargs):
            super(MaskedPooling, self).__init__(**kwargs)
            self.pooling_type = pooling_type
            
        def call(self, inputs, mask=None):
            if mask is not None:
                mask = tf.cast(mask, dtype=inputs.dtype)
                mask = tf.expand_dims(mask, axis=-1)
                
                if self.pooling_type == 'max':
                    return tf.reduce_max(inputs * mask - 1e10 * (1 - mask), axis=1)
                else:  # average
                    sum_pool = tf.reduce_sum(inputs * mask, axis=1)
                    count = tf.reduce_sum(mask, axis=1)
                    return sum_pool / (count + tf.keras.backend.epsilon())
            
            if self.pooling_type == 'max':
                return tf.reduce_max(inputs, axis=1)
            return tf.reduce_mean(inputs, axis=1)
    
    # LSTM with Skip Connections
    input_layer, mask_layer = get_input_layers()
    lstm1 = tf.keras.layers.LSTM(64, return_sequences=True)(mask_layer)
    lstm2 = tf.keras.layers.LSTM(64, return_sequences=False)(lstm1)
    skip_connection = MaskedPooling()(lstm1)
    merged = tf.keras.layers.Add()([lstm2, skip_connection])
    dense1 = tf.keras.layers.Dense(64, activation='tanh')(merged)
    dropout = tf.keras.layers.Dropout(0.2)(dense1)
    output = tf.keras.layers.Dense(num_classes, activation='softmax')(dropout)
    models["skip_lstm"] = tf.keras.Model(inputs=input_layer, outputs=output)
    
    # Parallel LSTM Processing
    input_layer, mask_layer = get_input_layers()
    lstm1 = tf.keras.layers.LSTM(32, return_sequences=False)(mask_layer)
    lstm2 = tf.keras.layers.LSTM(32, return_sequences=False)(mask_layer)
    concat = tf.keras.layers.Concatenate()([lstm1, lstm2])
    dense1 = tf.keras.layers.Dense(64, activation='tanh')(concat)
    dropout = tf.keras.layers.Dropout(0.2)(dense1)
    output = tf.keras.layers.Dense(num_classes, activation='softmax')(dropout)
    models["parallel_lstm"] = tf.keras.Model(inputs=input_layer, outputs=output)
    
    # Hierarchical Feature LSTM
    input_layer, mask_layer = get_input_layers()
    lstm1 = tf.keras.layers.LSTM(64, return_sequences=True)(mask_layer)
    pool1 = MaskedPooling(pooling_type='max')(lstm1)
    pool2 = MaskedPooling(pooling_type='avg')(lstm1)
    concat = tf.keras.layers.Concatenate()([pool1, pool2])
    dense1 = tf.keras.layers.Dense(64, activation='tanh')(concat)
    dropout = tf.keras.layers.Dropout(0.2)(dense1)
    output = tf.keras.layers.Dense(num_classes, activation='softmax')(dropout)
    models["hierarchical_lstm"] = tf.keras.Model(inputs=input_layer, outputs=output)
    
    return models


def evaluate_model(model, x_test, y_test, classes):
    """Evaluates model and returns metrics."""
    y_pred = model.predict(x_test)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true_classes = np.argmax(y_test, axis=1)
    
    # Calculate metrics
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true_classes, y_pred_classes, average='weighted'
    )
    
    # Create confusion matrix
    cm = confusion_matrix(y_true_classes, y_pred_classes)
    
    return {
        'precision': float(precision),
        'recall': float(recall),
        'f1': float(f1),
        'confusion_matrix': cm.tolist()
    }

def train_all_models(x_train, y_train, x_test, y_test, input_shape, num_classes, classes):
    """Trains and evaluates all model variants."""
    results = {}
    models = create_model_variants(input_shape, num_classes)
    
    for model_name, model in models.items():
        print(f"\nTraining {model_name}...")
        model_dir = os.path.join(BASE_DIR, model_name)
        os.makedirs(model_dir, exist_ok=True)
        
        # Compile model
        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        # Callbacks
        callbacks = [
            tf.keras.callbacks.ModelCheckpoint(
                os.path.join(model_dir, 'best_model.keras'),
                save_best_only=True,
                monitor='val_accuracy'
            ),
            tf.keras.callbacks.CSVLogger(
                os.path.join(model_dir, 'training_log.csv')
            )
        ]
        
        # Train model
        try:
            with tf.device('/CPU:0'):  # Force CPU usage
                history = model.fit(
                    x_train, y_train,
                    epochs=150,
                    batch_size=32,
                    validation_split=0.2,
                    callbacks=callbacks,
                    verbose=1
                )
            
            # Save final model
            model.save(os.path.join(model_dir, 'final_model.keras'))
            
            # Evaluate model
            metrics = evaluate_model(model, x_test, y_test, classes)
            
            # Save metrics and history
            results[model_name] = {
                'metrics': metrics,
                'history': {
                    'accuracy': history.history['accuracy'],
                    'val_accuracy': history.history['val_accuracy'],
                    'loss': history.history['loss'],
                    'val_loss': history.history['val_loss']
                }
            }
            
            # Plot and save confusion matrix
            plt.figure(figsize=(10, 8))
            ConfusionMatrixDisplay(
                confusion_matrix=np.array(metrics['confusion_matrix']),
                display_labels=classes
            ).plot()
            plt.title(f'Confusion Matrix - {model_name}')
            plt.savefig(os.path.join(model_dir, 'confusion_matrix.png'))
            plt.close()
            
        except Exception as e:
            print(f"Error training {model_name}: {str(e)}")
            results[model_name] = {'error': str(e)}
    
    # Save all results
    with open(os.path.join(BASE_DIR, 'results.json'), 'w') as f:
        json.dump(results, f, indent=4)
    
    return results

def create_benchmark_report(results):
    """Creates a benchmark report comparing all models."""
    report = []
    metrics = ['precision', 'recall', 'f1']
    
    for model_name, result in results.items():
        if 'error' in result:
            continue
        
        model_metrics = result['metrics']
        report.append({
            'model': model_name,
            **{metric: model_metrics[metric] for metric in metrics}
        })
    
    # Convert to DataFrame and sort by F1 score
    df = pd.DataFrame(report)
    df = df.sort_values('f1', ascending=False)
    
    # Save report
    df.to_csv(os.path.join(BASE_DIR, 'benchmark_report.csv'), index=False)
    
    return df
def main():
    """Main function to run the entire training and evaluation pipeline."""
    print("Starting ISL Training Pipeline...")
    
    # Step 1: Data Preprocessing
    print("\nPreprocessing data...")
    features, labels, classes = preprocess_data(CSV_PATH, quick_train=False)  # Set quick_train=False for full dataset
    
    # Step 2: Split the data
    print("\nSplitting data into train and test sets...")
    x_train, x_test, y_train, y_test = train_test_split(
        features, labels, test_size=0.2, random_state=42, stratify=np.argmax(labels, axis=1)
    )
    
    # Step 3: Define model parameters
    input_shape = (MAX_SEQUENCE_LENGTH, FEATURE_DIM)
    num_classes = len(classes)
    print(f"\nInput shape: {input_shape}")
    print(f"Number of classes: {num_classes}")
    
    # Step 4: Train and evaluate all models
    print("\nStarting model training and evaluation...")
    results = train_all_models(x_train, y_train, x_test, y_test, input_shape, num_classes, classes)
    
    # Step 5: Create and save benchmark report
    print("\nCreating benchmark report...")
    benchmark_df = create_benchmark_report(results)
    print("\nBenchmark Report:")
    print(benchmark_df)
    
    print(f"\nTraining complete. Results saved in: {BASE_DIR}")

if __name__ == "__main__":
    main()
